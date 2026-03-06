from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm import tqdm
from torchvision import transforms


@dataclass
class NSGAResult:
    original_label: int
    best_ce: float
    best_score: float
    class_preserved: bool
    history_best_ce: List[float]
    history_best_score: List[float]
    archive_ce_history: List[np.ndarray]
    archive_score_history: List[np.ndarray]
    archive_pred_history: List[np.ndarray]
    pareto_ce: np.ndarray
    pareto_score: np.ndarray
    pareto_pred: np.ndarray
    final_archive_perturbations: np.ndarray
    final_archive_adversarials: np.ndarray
    best_perturbation: np.ndarray
    best_adversarial: np.ndarray


class NSGABcosAttack:
    def __init__(
        self,
        model,
        device,
        num_iterations: int,
        population_size: int,
        epsilon: int,
        mutation_sigma: float = 0.35,
        crossover_alpha_min: float = 0.2,
        crossover_alpha_max: float = 0.8,
        score_mode: str = "mean",
        score_objective: str = "min",
    ):
        self.model = model
        self.device = device
        self.num_iterations = int(num_iterations)
        self.population_size = int(population_size)
        self.epsilon = int(epsilon)
        self.mutation_sigma = float(mutation_sigma)
        self.crossover_alpha_min = float(crossover_alpha_min)
        self.crossover_alpha_max = float(crossover_alpha_max)
        self.score_mode = str(score_mode)
        self.score_objective = str(score_objective)

        if self.population_size < 2:
            raise ValueError("population_size must be >= 2")
        if self.epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        if self.score_mode not in {"descriptor_1", "descriptor_2", "mean"}:
            raise ValueError("score_mode must be one of: 'descriptor_1', 'descriptor_2', 'mean'")
        if self.score_objective not in {"min", "max"}:
            raise ValueError("score_objective must be one of: 'min', 'max'")

        self.nds = NonDominatedSorting()
        self._split_transform()

    def _split_transform(self) -> None:
        bcos_transform_class = self.model.transform
        self.spatial_transform = transforms.Compose(
            bcos_transform_class.transforms.transforms[:-3]
        )
        self.bcos_transform = transforms.Compose(
            bcos_transform_class.transforms.transforms[-3:]
        )

    def _to_model_batch(self, adversarial_imgs: np.ndarray) -> torch.Tensor:
        imgs_pil = [Image.fromarray(img.astype(np.uint8)) for img in adversarial_imgs]
        return torch.stack([self.bcos_transform(img) for img in imgs_pil], dim=0).to(self.device)

    def evaluate(
        self,
        adversarial_imgs: np.ndarray,
        gt_explain_map: torch.Tensor,
        gt_contribution_map: torch.Tensor,
        gt_label: int,
    ) -> Dict[str, np.ndarray]:
        # checked

        model_batch = self._to_model_batch(adversarial_imgs).requires_grad_(True)
        explain_results = self.model.explain(model_batch, gt_label)

        logits = explain_results["logits"].to(self.device)
        explain_maps = explain_results["explanation"]
        contribution_maps = explain_results["contribution_map"]

        if not isinstance(explain_maps, torch.Tensor):
            explain_maps = torch.as_tensor(explain_maps)
        if not isinstance(contribution_maps, torch.Tensor):
            contribution_maps = torch.as_tensor(contribution_maps)
        if not isinstance(gt_explain_map, torch.Tensor):
            gt_explain_map = torch.as_tensor(gt_explain_map)
        if not isinstance(gt_contribution_map, torch.Tensor):
            gt_contribution_map = torch.as_tensor(gt_contribution_map)

        if gt_explain_map.dim() == 3:
            gt_explain_map = gt_explain_map.unsqueeze(0)
        if gt_contribution_map.dim() == 2:
            gt_contribution_map = gt_contribution_map.unsqueeze(0)

        explain_flat = explain_maps.reshape(explain_maps.shape[0], -1)
        gt_explain_flat = gt_explain_map.reshape(gt_explain_map.shape[0], -1)
        if gt_explain_flat.shape[0] == 1 and explain_flat.shape[0] > 1:
            gt_explain_flat = gt_explain_flat.expand(explain_flat.shape[0], -1)
        descriptor_1 = F.cosine_similarity(explain_flat, gt_explain_flat, dim=1, eps=1e-8).cpu()

        contribution_flat = contribution_maps.reshape(contribution_maps.shape[0], -1)
        gt_contribution_flat = gt_contribution_map.reshape(gt_contribution_map.shape[0], -1)
        if gt_contribution_flat.shape[0] == 1 and contribution_flat.shape[0] > 1:
            gt_contribution_flat = gt_contribution_flat.expand(contribution_flat.shape[0], -1)
        descriptor_2 = F.cosine_similarity(contribution_flat, gt_contribution_flat, dim=1, eps=1e-8)
        descriptor_2 = torch.clamp(descriptor_2, 0.0, 1.0).cpu()

        if self.score_mode == "descriptor_1":
            score = descriptor_1
        elif self.score_mode == "descriptor_2":
            score = descriptor_2
        elif self.score_mode == "mean":
            score = 0.5 * (descriptor_1 + descriptor_2)
        else:
            raise ValueError(f"Unsupported score_mode: {self.score_mode}")

        targets = torch.full((logits.shape[0],), int(gt_label), dtype=torch.long, device=self.device)
        ce = F.cross_entropy(logits, targets, reduction="none")

        preds = logits.argmax(dim=1)

        return {
            "ce": ce.detach().cpu().numpy().copy(),
            "score": score.detach().cpu().numpy().copy(),
            "descriptor_1": descriptor_1.detach().cpu().numpy().copy(),
            "descriptor_2": descriptor_2.detach().cpu().numpy().copy(),
            "pred": preds.detach().cpu().numpy().copy(),
        }

    def _sample_parent_pairs(self, pool_size: int) -> Tuple[np.ndarray, np.ndarray]:
        # checked
        idx_a = np.random.choice(pool_size, size=self.population_size, replace=True)
        non_zero_offsets = np.random.choice(np.arange(1, pool_size), size=self.population_size, replace=True)
        idx_b = (idx_a + non_zero_offsets) % pool_size
        return idx_a.astype(np.int64, copy=True), idx_b.astype(np.int64, copy=True)

    def _crossover(self, noise_a: np.ndarray, noise_b: np.ndarray) -> np.ndarray:
        # checked
        alpha = np.random.uniform(
            -self.epsilon,
            self.epsilon,
            size=(noise_a.shape[0], 1, 1, 1),
        ).astype(np.float32)
        return np.clip(alpha * noise_a + (1.0 - alpha) * noise_b, -self.epsilon, self.epsilon).astype(np.float32, copy=True)

    def _mutate(self, offspring_noises: np.ndarray) -> np.ndarray:
        # checked
        mutation_prob = float(np.clip(self.mutation_sigma, 0.0, 1.0))
        random_perturbations = np.random.randint(
            low=-self.epsilon,
            high=self.epsilon + 1,
            size=offspring_noises.shape,
            dtype=np.int16,
        )
        resample_mask = np.random.rand(offspring_noises.shape[0], 1, 1, 1) < mutation_prob
        mutated = np.where(resample_mask, random_perturbations, offspring_noises)
        return np.clip(mutated, -self.epsilon, self.epsilon).astype(np.int16).copy()

    def _build_adversarial_images(self, base_img: np.ndarray, perturbations: np.ndarray) -> np.ndarray:
        # Checked
        
        return np.clip(
            base_img[None, ...].astype(np.int16) + perturbations,
            0,
            255,
        ).astype(np.uint8).copy()


    def _to_objectives(self, metrics: Dict[str, np.ndarray]) -> np.ndarray:
        obj_ce = metrics["ce"]
        if self.score_objective == "min":
            obj_score = metrics["score"]
        else:
            obj_score = -metrics["score"]
        return np.column_stack([obj_ce, obj_score]).astype(np.float64, copy=True)

    @staticmethod
    def _crowding_distance(F: np.ndarray) -> np.ndarray:
        infinity = 1e+14

        n_points = F.shape[0]
        n_obj = F.shape[1]

        if n_points <= 2:
            return np.full(n_points, infinity)
        else:

            # sort each column and get index
            I = np.argsort(F, axis=0, kind='mergesort')

            # now really sort the whole array
            F = F[I, np.arange(n_obj)]

            # get the distance to the last element in sorted list and replace zeros with actual values
            dist = np.concatenate([F, np.full((1, n_obj), np.inf)]) - np.concatenate([np.full((1, n_obj), -np.inf), F])

            index_dist_is_zero = np.where(dist == 0)

            dist_to_last = np.copy(dist)
            for i, j in zip(*index_dist_is_zero):
                dist_to_last[i, j] = dist_to_last[i - 1, j]

            dist_to_next = np.copy(dist)
            for i, j in reversed(list(zip(*index_dist_is_zero))):
                dist_to_next[i, j] = dist_to_next[i + 1, j]

            # normalize all the distances
            norm = np.max(F, axis=0) - np.min(F, axis=0)
            norm[norm == 0] = np.nan
            dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

            # if we divided by zero because all values in one columns are equal replace by none
            dist_to_last[np.isnan(dist_to_last)] = 0.0
            dist_to_next[np.isnan(dist_to_next)] = 0.0

            # sum up the distance to next and last and norm by objectives - also reorder from sorted list
            J = np.argsort(I, axis=0)
            crowding = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # replace infinity with a large number
        crowding[np.isinf(crowding)] = infinity
        return crowding

    def _select_survivors(self, objectives: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        fronts = self.nds.do(objectives, n_stop_if_ranked=self.population_size)
        survivors: List[int] = []

        for front in fronts:
            front = np.asarray(front, dtype=np.int64)
            if len(front) == 0:
                continue

            if len(survivors) + len(front) <= self.population_size:
                survivors.extend(front.tolist())
                continue

            front_obj = objectives[front]
            crowd = self._crowding_distance(front_obj)
            sorted_front = front[np.argsort(-crowd)]
            needed = self.population_size - len(survivors)
            survivors.extend(sorted_front[:needed].tolist())
            break

        return np.asarray(survivors, dtype=np.int64), [np.asarray(f, dtype=np.int64) for f in fronts]

    def _pick_best_index(self, metrics: Dict[str, np.ndarray], original_label: int) -> int:
        # Final preference: keep original class and minimize score.
        same_class = np.where(metrics["pred"] == int(original_label))[0]
        if len(same_class) > 0:
            same_scores = metrics["score"][same_class]
            best_score = np.min(same_scores)
            score_ties = same_class[np.where(np.isclose(same_scores, best_score))[0]]
            tie_ce = metrics["ce"][score_ties]
            return int(score_ties[np.argmin(tie_ce)])

        # Fallback: if no sample keeps class, still minimize score first.
        best_score = np.min(metrics["score"])
        score_ties = np.where(np.isclose(metrics["score"], best_score))[0]
        tie_ce = metrics["ce"][score_ties]
        return int(score_ties[np.argmin(tie_ce)])

    def _current_archive_indices(self, metrics: Dict[str, np.ndarray]) -> np.ndarray:
        # checked: lấy ra rank 0
        objectives = self._to_objectives(metrics)
        fronts = self.nds.do(objectives, n_stop_if_ranked=objectives.shape[0])
        return np.asarray(fronts[0], dtype=np.int64)

    def run(self, img: Image.Image) -> NSGAResult:

        # repair for Attack
        img = self.spatial_transform(img) # PIL Image
        base_img = np.asarray(img, dtype=np.uint8).copy() # np image, shape: (H, W, C)

        original_tensor = self.bcos_transform(img).unsqueeze(0).to(self.device).requires_grad_(True) # tensor
        original_results = self.model.explain(original_tensor)
        original_explain_map = original_results["explanation"]
        original_contribution_map = original_results["contribution_map"]

        prediction = original_results["prediction"]
        if isinstance(prediction, torch.Tensor):
            original_label = int(prediction.reshape(-1)[0].item())
        elif isinstance(prediction, (list, tuple, np.ndarray)):
            original_label = int(prediction[0])
        else:
            original_label = int(prediction)

        # init population with random nois: int in [-epsilon, epsilon]
        population_noise = np.random.randint(
            low=-self.epsilon,
            high=self.epsilon + 1,
            size=(self.population_size, *base_img.shape),
            dtype=np.int16,
        ).copy() # shape: (population_size, H, W, C)
        population_images = self._build_adversarial_images(base_img, population_noise)
        metrics = self.evaluate(
            population_images,
            original_explain_map,
            original_contribution_map,
            original_label,
        )

        best_ce = float(np.min(metrics["ce"]))
        best_score = float(np.min(metrics["score"]))
        history_best_ce = [best_ce]
        history_best_score = [best_score]
        archive_ce_history: List[np.ndarray] = []
        archive_score_history: List[np.ndarray] = []
        archive_pred_history: List[np.ndarray] = []

        initial_archive_indices = self._current_archive_indices(metrics)
        archive_ce_history.append(metrics["ce"][initial_archive_indices].copy())
        archive_score_history.append(metrics["score"][initial_archive_indices].copy())
        archive_pred_history.append(metrics["pred"][initial_archive_indices].copy())

        any_class_preserved = bool(np.any(metrics["pred"] == int(original_label)))

        for _ in tqdm(range(self.num_iterations)):
            parent_idx_a, parent_idx_b = self._sample_parent_pairs(pool_size=population_noise.shape[0])
            noise_a = population_noise[parent_idx_a].astype(np.float32, copy=True)
            noise_b = population_noise[parent_idx_b].astype(np.float32, copy=True)

            offspring_noises = self._crossover(noise_a, noise_b)
            offspring_noises = self._mutate(offspring_noises)
            offspring_images = self._build_adversarial_images(base_img, offspring_noises)
            offspring_metrics = self.evaluate(
                offspring_images,
                original_explain_map,
                original_contribution_map,
                original_label,
            )

            merged_noise = np.concatenate([population_noise, offspring_noises], axis=0).copy()
            merged_images = np.concatenate([population_images, offspring_images], axis=0).copy()
            merged_metrics = {
                key: np.concatenate([metrics[key], offspring_metrics[key]], axis=0).copy()
                for key in metrics.keys()
            }

            objectives = self._to_objectives(merged_metrics)
            survivor_indices, _ = self._select_survivors(objectives)

            population_noise = merged_noise[survivor_indices].copy()
            population_images = merged_images[survivor_indices].copy()
            metrics = {key: value[survivor_indices].copy() for key, value in merged_metrics.items()}

            archive_indices = self._current_archive_indices(metrics)
            archive_ce_history.append(metrics["ce"][archive_indices].copy())
            archive_score_history.append(metrics["score"][archive_indices].copy())
            archive_pred_history.append(metrics["pred"][archive_indices].copy())
            any_class_preserved = any_class_preserved or bool(np.any(metrics["pred"] == int(original_label)))

            current_min_ce = float(np.min(metrics["ce"]))
            current_min_score = float(np.min(metrics["score"]))
            best_ce = min(best_ce, current_min_ce)
            best_score = min(best_score, current_min_score)

            history_best_ce.append(best_ce)
            history_best_score.append(best_score)

        pareto_indices = self._current_archive_indices(metrics)
        final_best_idx = self._pick_best_index(metrics, original_label)
        best_perturbation = population_noise[final_best_idx].copy()
        best_adversarial = population_images[final_best_idx].copy()

        class_preserved = any_class_preserved

        return NSGAResult(
            original_label=original_label,
            best_ce=best_ce,
            best_score=best_score,
            class_preserved=class_preserved,
            history_best_ce=[float(v) for v in history_best_ce],
            history_best_score=[float(v) for v in history_best_score],
            archive_ce_history=archive_ce_history,
            archive_score_history=archive_score_history,
            archive_pred_history=archive_pred_history,
            pareto_ce=metrics["ce"][pareto_indices].copy(),
            pareto_score=metrics["score"][pareto_indices].copy(),
            pareto_pred=metrics["pred"][pareto_indices].copy(),
            final_archive_perturbations=population_noise[pareto_indices].copy(),
            final_archive_adversarials=population_images[pareto_indices].copy(),
            best_perturbation=best_perturbation,
            best_adversarial=best_adversarial,
        )
