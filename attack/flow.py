from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

@dataclass
class Elite:
    ce: float
    descriptor_1: float
    descriptor_2: float
    pred: int
    image: np.ndarray
    perturbation: np.ndarray


class EliteMap:
    def __init__(
        self,
        descriptor_1_bins: int,
        descriptor_2_bins: int,
        descriptor_1_max: float,
        descriptor_1_min: float,
        descriptor_2_max: float,
        descriptor_2_min: float,
    ):
        if descriptor_1_bins <= 0 or descriptor_2_bins <= 0:
            raise ValueError("Bins must be > 0")
        if descriptor_1_max <= descriptor_1_min:
            raise ValueError("descriptor_1_max must be > descriptor_1_min")
        if descriptor_2_max <= descriptor_2_min:
            raise ValueError("descriptor_2_max must be > descriptor_2_min")

        self.descriptor_1_bins = int(descriptor_1_bins)
        self.descriptor_2_bins = int(descriptor_2_bins)
        self.descriptor_1_max = float(descriptor_1_max)
        self.descriptor_1_min = float(descriptor_1_min)
        self.descriptor_2_max = float(descriptor_2_max)
        self.descriptor_2_min = float(descriptor_2_min)

        self.quality_ce = np.full((self.descriptor_1_bins, self.descriptor_2_bins), np.inf, dtype=np.float32)
        self.descriptor_1 = np.zeros((self.descriptor_1_bins, self.descriptor_2_bins), dtype=np.float32)
        self.descriptor_2 = np.zeros((self.descriptor_1_bins, self.descriptor_2_bins), dtype=np.float32)
        self.pred = np.full((self.descriptor_1_bins, self.descriptor_2_bins), -1, dtype=np.int32)
        self.images: Dict[Tuple[int, int], np.ndarray] = {}
        self.perturbations: Dict[Tuple[int, int], np.ndarray] = {}

    def _cell_indices(self, descriptor_1_value: float, descriptor_2_value: float) -> Tuple[int, int]:
        descriptor_1_ratio = (descriptor_1_value - self.descriptor_1_min) / max(
            self.descriptor_1_max - self.descriptor_1_min,
            1e-12,
        )
        descriptor_2_ratio = (descriptor_2_value - self.descriptor_2_min) / max(
            self.descriptor_2_max - self.descriptor_2_min,
            1e-12,
        )
        descriptor_1_ratio = min(max(descriptor_1_ratio, 0.0), 1.0)
        descriptor_2_ratio = min(max(descriptor_2_ratio, 0.0), 1.0)
        descriptor_1_idx = min(self.descriptor_1_bins - 1, int(descriptor_1_ratio * self.descriptor_1_bins))
        descriptor_2_idx = min(self.descriptor_2_bins - 1, int(descriptor_2_ratio * self.descriptor_2_bins))
        return descriptor_1_idx, descriptor_2_idx

    def try_insert(self, elite: Elite) -> bool:
        if not (
            np.isfinite(elite.ce)
            and np.isfinite(elite.descriptor_1)
            and np.isfinite(elite.descriptor_2)
        ):
            return False

        d1_idx, d2_idx = self._cell_indices(elite.descriptor_1, elite.descriptor_2)
        if elite.ce >= float(self.quality_ce[d1_idx, d2_idx]):
            return False

        self.quality_ce[d1_idx, d2_idx] = float(elite.ce)
        self.descriptor_1[d1_idx, d2_idx] = float(elite.descriptor_1)
        self.descriptor_2[d1_idx, d2_idx] = float(elite.descriptor_2)
        self.pred[d1_idx, d2_idx] = int(elite.pred)
        self.images[(d1_idx, d2_idx)] = elite.image.copy()
        self.perturbations[(d1_idx, d2_idx)] = elite.perturbation.copy()
        return True

    def occupied_cells(self) -> int:
        return len(self.perturbations)

    def occupied_indices(self) -> List[Tuple[int, int]]:
        return list(self.images.keys())

    def sample_images(self, n: int, fallback: np.ndarray) -> np.ndarray:
        occupied = self.occupied_indices()
        if len(occupied) == 0:
            return np.repeat(fallback[None, ...], n, axis=0).copy()
        sampled_keys = np.random.choice(len(occupied), size=n, replace=True)
        return np.stack([self.images[occupied[idx]] for idx in sampled_keys], axis=0).copy()

    def sample_perturbations(self, n: int, fallback: np.ndarray) -> np.ndarray:
        occupied = list(self.perturbations.keys())
        if len(occupied) == 0:
            return np.repeat(fallback[None, ...], n, axis=0).copy()
        sampled_keys = np.random.choice(len(occupied), size=n , replace=True)
        return np.stack([self.perturbations[occupied[idx]] for idx in sampled_keys], axis=0).copy()


class Attack:
    def __init__(
        self,
        model,
        device,
        num_iterations: int,
        population_size: int,
        epsilon: int,
        elite_map: EliteMap,
        mutation_sigma: float = 0.35,
        parent_source: str = "population",
    ):
        self.model = model
        self.device = device
        self.num_iterations = int(num_iterations)
        self.population_size = int(population_size)
        self.epsilon = int(epsilon)
        self.elite_map = elite_map
        self.mutation_sigma = float(mutation_sigma)
        self.parent_source = str(parent_source)
        self.split_transform()
        if self.parent_source not in {"population", "elite_map"}:
            raise ValueError("parent_source must be one of: 'population', 'elite_map'")

    def split_transform(self):
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
        base_img: np.ndarray,
    ) -> Dict[str, np.ndarray]:
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

        targets = torch.full((logits.shape[0],), int(gt_label), dtype=torch.long, device=self.device)
        ce = F.cross_entropy(logits, targets, reduction="none")

        if gt_explain_map.dim() == 3:
            gt_explain_map = gt_explain_map.unsqueeze(0)
        if gt_contribution_map.dim() == 2:
            gt_contribution_map = gt_contribution_map.unsqueeze(0)

        explain_flat = explain_maps.reshape(explain_maps.shape[0], -1)
        gt_explain_flat = gt_explain_map.reshape(gt_explain_map.shape[0], -1)
        if gt_explain_flat.shape[0] == 1 and explain_flat.shape[0] > 1:
            gt_explain_flat = gt_explain_flat.expand(explain_flat.shape[0], -1)
        explain_map_similarity = F.cosine_similarity(explain_flat, gt_explain_flat, dim=1, eps=1e-8)
        # explain_map_similarity = torch.log(1 + torch.clamp(explain_map_similarity, 0, 1.0))

        contribution_flat = contribution_maps.reshape(contribution_maps.shape[0], -1)
        gt_contribution_flat = gt_contribution_map.reshape(gt_contribution_map.shape[0], -1)
        if gt_contribution_flat.shape[0] == 1 and contribution_flat.shape[0] > 1:
            gt_contribution_flat = gt_contribution_flat.expand(contribution_flat.shape[0], -1)
        contribution_map_similarity = F.cosine_similarity(contribution_flat, gt_contribution_flat, dim=1, eps=1e-8)
        contribution_map_similarity = torch.clamp(contribution_map_similarity, 0, 1.0)
        # contribution_map_similarity = torch.log(1 + contribution_map_similarity)

        preds = logits.argmax(dim=1)
        return {
            "ce": ce.detach().cpu().numpy().copy(),
            "descriptor_1": explain_map_similarity.detach().cpu().numpy().copy(),
            "descriptor_2": contribution_map_similarity.detach().cpu().numpy().copy(),
            "pred": preds.detach().cpu().numpy().copy(),
        }


    def _sample_parent_pairs(self, pool_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if pool_size < 2:
            raise ValueError("pool_size must be >= 2")

        parent_idx_a = np.random.choice(pool_size, size=self.population_size, replace=True)
        non_zero_offsets = np.random.choice(np.arange(1, pool_size), size=self.population_size, replace=True)
        parent_idx_b = (parent_idx_a + non_zero_offsets) % pool_size
        return parent_idx_a.astype(np.int64, copy=True), parent_idx_b.astype(np.int64, copy=True)
    

    def _crossover_noises(self, noise_a: np.ndarray, noise_b: np.ndarray) -> np.ndarray:
        alpha = np.random.uniform(0.2, 0.8, size=(noise_a.shape[0], 1, 1, 1)).astype(np.float32)
        return (alpha * noise_a + (1.0 - alpha) * noise_b).copy()

    def _mutate_noises(self, offspring_noises: np.ndarray) -> np.ndarray:
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

    def _insert_batch(
        self,
        adversarial_imgs: np.ndarray,
        perturbations: np.ndarray,
        metrics: Dict[str, np.ndarray],
    ) -> int:
        inserted = 0
        for idx in range(adversarial_imgs.shape[0]):
            if self.elite_map.try_insert(
                Elite(
                    ce=float(metrics["ce"][idx]),
                    descriptor_1=float(metrics["descriptor_1"][idx]),
                    descriptor_2=float(metrics["descriptor_2"][idx]),
                    pred=int(metrics["pred"][idx]),
                    image=adversarial_imgs[idx],
                    perturbation=perturbations[idx],
                )
            ):
                inserted += 1
        return inserted

    def tournament_selection(self, ce_pool: np.ndarray, tournament_size: int = 4) -> np.ndarray:
        if len(ce_pool) == 0:
            raise ValueError("ce_pool must not be empty")

        tournament_size = max(2, int(tournament_size))
        selected = []
        for _ in range(self.population_size):
            indices = np.random.choice(len(ce_pool), size=min(tournament_size, len(ce_pool)), replace=False)
            best_local = indices[np.argmin(ce_pool[indices])]
            selected.append(int(best_local))
        return np.asarray(selected, dtype=np.int64)


    def run(self, img: Image.Image) -> Dict[str, object]:
        img = self.spatial_transform(img) # PIL image
        base_img = np.asarray(img, dtype=np.uint8).copy()

        original_tensor = self.bcos_transform(img).unsqueeze(0).to(self.device).requires_grad_(True)
        original_results = self.model.explain(original_tensor)
        original_explain_map = original_results["explanation"]
        original_contribution_map = original_results["contribution_map"]
        base_model_input = original_tensor[0].detach()
        prediction = original_results["prediction"]
        if isinstance(prediction, torch.Tensor):
            original_label = int(prediction.reshape(-1)[0].item())
        elif isinstance(prediction, (list, tuple, np.ndarray)):
            original_label = int(prediction[0])
        else:
            original_label = int(prediction)

        print(f"Original prediction: {original_label}")

        population_noise = np.random.randint(
            low=-self.epsilon,
            high=self.epsilon + 1,
            size=(self.population_size, *base_img.shape),
            dtype=np.int16,
        ).copy()

        adversarial_images = np.clip(base_img[None, ...].astype(np.int16) + population_noise, 0, 255).astype(np.uint8).copy()
        metrics = self.evaluate(
            adversarial_images,
            original_explain_map,
            original_contribution_map,
            original_label,
            base_img,
        )
        self._insert_batch(adversarial_images, population_noise, metrics)

        best_ce = float(np.min(metrics["ce"]))
        history_best_ce = [best_ce]

        for iteration in tqdm(range(self.num_iterations)):
            if self.parent_source == "population":
                parent_pool = population_noise.copy()
            else:
                elite_keys = list(self.elite_map.perturbations.keys())
                elite_candidate_count = len(elite_keys)
                if elite_candidate_count == 0:
                    parent_pool = population_noise.copy()
                else:
                    elite_candidates = [self.elite_map.perturbations[key] for key in elite_keys]
                    quality_values = np.asarray(
                        [float(self.elite_map.quality_ce[d1_idx, d2_idx]) for (d1_idx, d2_idx) in elite_keys],
                        dtype=np.float64,
                    )
                    quality_values = np.nan_to_num(quality_values, nan=0.0, posinf=0.0, neginf=0.0)

                    min_quality = float(np.min(quality_values))
                    max_quality = float(np.max(quality_values))
                    if max_quality > min_quality:
                        weights = (quality_values - min_quality) / (max_quality - min_quality)
                        weights = weights + 1e-3
                        weights = weights / np.sum(weights)
                        sampled_indices = np.random.choice(
                            elite_candidate_count,
                            size=self.population_size,
                            replace=True,
                            p=weights,
                        )
                    else:
                        sampled_indices = np.random.choice(
                            elite_candidate_count,
                            size=self.population_size,
                            replace=True,
                        )
                    parent_pool = np.stack([elite_candidates[idx] for idx in sampled_indices], axis=0).astype(np.int16, copy=True)

            parent_idx_a, parent_idx_b = self._sample_parent_pairs(pool_size=parent_pool.shape[0])
            noise_a = parent_pool[parent_idx_a].astype(np.float32, copy=True)
            noise_b = parent_pool[parent_idx_b].astype(np.float32, copy=True)
            offspring_noises = self._crossover_noises(noise_a, noise_b)
            offspring_noises = self._mutate_noises(offspring_noises)

            adversarial_images = np.clip(
                base_img[None, ...].astype(np.int16) + offspring_noises,
                0,
                255,
            ).astype(np.uint8).copy()
            offspring_metrics = self.evaluate(
                adversarial_images,
                original_explain_map,
                original_contribution_map,
                original_label,
                base_img,
            )
            self._insert_batch(adversarial_images, offspring_noises, offspring_metrics)

            if self.parent_source == "population":
                ce_pool = np.concatenate([metrics["ce"], offspring_metrics["ce"]], axis=0).copy()
                descriptor_1_pool = np.concatenate([metrics["descriptor_1"], offspring_metrics["descriptor_1"]], axis=0).copy()
                descriptor_2_pool = np.concatenate([metrics["descriptor_2"], offspring_metrics["descriptor_2"]], axis=0).copy()
                pred_pool = np.concatenate([metrics["pred"], offspring_metrics["pred"]], axis=0).copy()
                population_noises = np.concatenate([population_noise, offspring_noises], axis=0).copy()
                population_images = np.concatenate([base_img[None, ...].astype(np.int16) + population_noise, base_img[None, ...].astype(np.int16) + offspring_noises], axis=0)
                population_images = np.clip(population_images, 0, 255).astype(np.uint8).copy()

                selected_indices = self.tournament_selection(ce_pool, tournament_size=4)
                population_noise = population_noises[selected_indices].copy()
                adversarial_images = population_images[selected_indices].copy()
                metrics = {
                    "ce": ce_pool[selected_indices].copy(),
                    "descriptor_1": descriptor_1_pool[selected_indices].copy(),
                    "descriptor_2": descriptor_2_pool[selected_indices].copy(),
                    "pred": pred_pool[selected_indices].copy(),
                }
            else:
                population_noise = offspring_noises.copy()
                metrics = {
                    "ce": offspring_metrics["ce"].copy(),
                    "descriptor_1": offspring_metrics["descriptor_1"].copy(),
                    "descriptor_2": offspring_metrics["descriptor_2"].copy(),
                    "pred": offspring_metrics["pred"].copy(),
                }
            print(metrics)

            # print(metrics)
            iter_best_ce = float(np.min(metrics["ce"]))
            best_ce = min(best_ce, iter_best_ce)
            history_best_ce.append(best_ce)

            print(
                f"Iter {iteration + 1}/{self.num_iterations}, "
                f"Best CE: {best_ce:.4f}, "
                f"Iter Best CE: {iter_best_ce:.4f}, "
                f"Occupied Cells: {self.elite_map.occupied_cells()}"
            )

        return {
            "original_label": original_label,
            "best_quality_ce": best_ce,
            "occupied_cells": self.elite_map.occupied_cells(),
            "history_best_ce": history_best_ce,
            "quality_ce": self.elite_map.quality_ce.copy(),
            "descriptor_1": self.elite_map.descriptor_1.copy(),
            "descriptor_2": self.elite_map.descriptor_2.copy(),
            "pred": self.elite_map.pred.copy(),
            "perturbations": {cell: perturb.copy() for cell, perturb in self.elite_map.perturbations.items()},
        }


