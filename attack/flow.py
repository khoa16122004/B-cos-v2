from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

@dataclass
class Elite:
    ce: float
    l2: float
    diff_explain: float
    pred: int
    image: np.ndarray
    perturbation: np.ndarray


class EliteMap:
    def __init__(
        self,
        explain_diff_bins: int,
        l2_diff_bins: int,
        l2_max: float,
        explain_diff_max: float = 1.0,
    ):
        if explain_diff_bins <= 0 or l2_diff_bins <= 0:
            raise ValueError("Bins must be > 0")
        if l2_max <= 0:
            raise ValueError("l2_max must be > 0")

        self.explain_diff_bins = explain_diff_bins
        self.l2_diff_bins = l2_diff_bins
        self.l2_max = float(l2_max)
        self.explain_diff_max = float(explain_diff_max)

        self.quality_ce = np.full((explain_diff_bins, l2_diff_bins), np.inf, dtype=np.float32)
        self.l2 = np.zeros((explain_diff_bins, l2_diff_bins), dtype=np.float32)
        self.diff_explain = np.zeros((explain_diff_bins, l2_diff_bins), dtype=np.float32)
        self.pred = np.full((explain_diff_bins, l2_diff_bins), -1, dtype=np.int32)
        self.images: Dict[Tuple[int, int], np.ndarray] = {}
        self.perturbations: Dict[Tuple[int, int], np.ndarray] = {}

    def _cell_indices(self, l2_value: float, diff_explain_value: float) -> Tuple[int, int]:
        l2_ratio = min(max(l2_value / max(self.l2_max, 1e-12), 0.0), 1.0)
        diff_ratio = min(max(diff_explain_value / max(self.explain_diff_max, 1e-12), 0.0), 1.0)
        diff_idx = min(self.explain_diff_bins - 1, int(diff_ratio * self.explain_diff_bins))
        l2_idx = min(self.l2_diff_bins - 1, int(l2_ratio * self.l2_diff_bins))
        return diff_idx, l2_idx

    def try_insert(self, elite: Elite) -> bool:
        d_idx, l_idx = self._cell_indices(elite.l2, elite.diff_explain)
        if elite.ce >= float(self.quality_ce[d_idx, l_idx]):
            return False

        self.quality_ce[d_idx, l_idx] = float(elite.ce)
        self.l2[d_idx, l_idx] = float(elite.l2)
        self.diff_explain[d_idx, l_idx] = float(elite.diff_explain)
        self.pred[d_idx, l_idx] = int(elite.pred)
        self.images[(d_idx, l_idx)] = elite.image.copy()
        self.perturbations[(d_idx, l_idx)] = elite.perturbation.copy()
        return True

    def occupied_cells(self) -> int:
        return int(np.isfinite(self.quality_ce).sum())

    def occupied_indices(self) -> List[Tuple[int, int]]:
        return list(self.images.keys())

    def sample_images(self, n: int, fallback: np.ndarray) -> np.ndarray:
        occupied = self.occupied_indices()
        if len(occupied) == 0:
            return np.repeat(fallback[None, ...], n, axis=0)
        sampled_keys = np.random.choice(len(occupied), size=n, replace=True)
        return np.stack([self.images[occupied[idx]] for idx in sampled_keys], axis=0)


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
    ):
        self.model = model
        self.device = device
        self.num_iterations = int(num_iterations)
        self.population_size = int(population_size)
        self.epsilon = int(epsilon)
        self.elite_map = elite_map
        self.mutation_sigma = float(mutation_sigma)

    def _to_model_batch(self, adversarial_imgs: np.ndarray) -> torch.Tensor:
        imgs_pil = [Image.fromarray(img.astype(np.uint8)) for img in adversarial_imgs]
        return torch.stack([self.model.transform(img) for img in imgs_pil], dim=0).to(self.device)

    def evaluate(
        self,
        adversarial_imgs: np.ndarray,
        gt_map: torch.Tensor,
        gt_label: int,
        base_img: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        model_batch = self._to_model_batch(adversarial_imgs).requires_grad_(True)
        explain_results = self.model.explain(model_batch)
        logits = explain_results["logits"].to(self.device)
        maps = explain_results["explanation"]
        if not isinstance(maps, torch.Tensor):
            maps = torch.as_tensor(maps)
        maps = maps.to(self.device, dtype=model_batch.dtype)
        if not isinstance(gt_map, torch.Tensor):
            gt_map = torch.as_tensor(gt_map)
        gt_map = gt_map.to(self.device, dtype=maps.dtype)
        targets = torch.full((logits.shape[0],), int(gt_label), dtype=torch.long, device=self.device)
        ce = F.cross_entropy(logits, targets, reduction="none")
        
        maps_flat = maps.reshape(maps.shape[0], -1)
        gt_map_flat = gt_map.reshape(1, -1).expand(maps.shape[0], -1)
        cos_sim = F.cosine_similarity(maps_flat, gt_map_flat, dim=1, eps=1e-8).clamp(-1.0, 1.0)
        diff_explain = 1.0 - cos_sim
        deltas = (
            adversarial_imgs.astype(np.float32)
            - base_img.astype(np.float32)
        ) / 255.0

        l2 = np.sqrt(np.sum(deltas ** 2, axis=(1, 2, 3)))
        l2_max = np.sqrt(base_img.size)
        l2_scaled = l2 / l2_max
        l2_scaled = np.clip(l2_scaled, 0.0, 1.0)

        preds = logits.argmax(dim=1)
        return {
            "ce": ce.detach().cpu().numpy(),
            "diff_explain": diff_explain.detach().cpu().numpy(),
            "l2": l2_scaled,
            "pred": preds.detach().cpu().numpy(),
        }

    def _sample_parent_pairs(self, pool_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if pool_size < 2:
            raise ValueError("pool_size must be >= 2")

        if self.population_size <= pool_size:
            parent_idx_a = np.random.permutation(pool_size)[: self.population_size]
            parent_idx_b = np.random.permutation(pool_size)[: self.population_size]
            collision = parent_idx_a == parent_idx_b
            if np.any(collision):
                parent_idx_b[collision] = (parent_idx_b[collision] + 1) % pool_size
        else:
            parent_idx_a = np.random.randint(0, pool_size, size=self.population_size)
            offsets = np.random.randint(1, pool_size, size=self.population_size)
            parent_idx_b = (parent_idx_a + offsets) % pool_size

        return parent_idx_a.astype(np.int64), parent_idx_b.astype(np.int64)

    def _crossover_noises(self, noise_a: np.ndarray, noise_b: np.ndarray) -> np.ndarray:
        alpha = np.random.uniform(0.2, 0.8, size=(noise_a.shape[0], 1, 1, 1)).astype(np.float32)
        return alpha * noise_a + (1.0 - alpha) * noise_b

    def _mutate_noises(self, offspring_noises: np.ndarray) -> np.ndarray:
        mut_sigma = max(1.0, self.epsilon * self.mutation_sigma)
        gaussian = np.random.normal(0.0, mut_sigma, size=offspring_noises.shape).astype(np.float32)
        return np.clip(offspring_noises + gaussian, -self.epsilon, self.epsilon).astype(np.int16)

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
                    l2=float(metrics["l2"][idx]),
                    diff_explain=float(metrics["diff_explain"][idx]),
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
        base_img = np.asarray(img).astype(np.uint8)

        original_tensor = self.model.transform(img).unsqueeze(0).to(self.device).requires_grad_(True)
        original_results = self.model.explain(original_tensor)
        original_map = original_results["explanation"]
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
        )
        adversarial_images = np.clip(base_img[None, ...].astype(np.int16) + population_noise, 0, 255).astype(np.uint8)

        metrics = self.evaluate(adversarial_images, original_map, original_label, base_img)
        self._insert_batch(adversarial_images, population_noise, metrics)

        best_ce = float(np.min(metrics["ce"]))
        history_best_ce = [best_ce]
        population_noise = adversarial_images.astype(np.int16) - base_img[None, ...].astype(np.int16)

        for iteration in tqdm(range(self.num_iterations)):
            parent_idx_a, parent_idx_b = self._sample_parent_pairs(pool_size=population_noise.shape[0])
            noise_a = population_noise[parent_idx_a].astype(np.float32)
            noise_b = population_noise[parent_idx_b].astype(np.float32)
            offspring_noises = self._crossover_noises(noise_a, noise_b)
            offspring_noises = self._mutate_noises(offspring_noises)

            adversarial_images = np.clip(
                base_img[None, ...].astype(np.int16) + offspring_noises,
                0,
                255,
            ).astype(np.uint8)
            offspring_metrics = self.evaluate(adversarial_images, original_map, original_label, base_img)
            self._insert_batch(adversarial_images, offspring_noises, offspring_metrics)

            ce_pool = np.concatenate([metrics["ce"], offspring_metrics["ce"]], axis=0)
            diff_pool = np.concatenate([metrics["diff_explain"], offspring_metrics["diff_explain"]], axis=0)
            l2_pool = np.concatenate([metrics["l2"], offspring_metrics["l2"]], axis=0)
            pred_pool = np.concatenate([metrics["pred"], offspring_metrics["pred"]], axis=0)
            population_noises = np.concatenate([population_noise, offspring_noises], axis=0)
            population_images = np.concatenate([base_img[None, ...].astype(np.int16) + population_noise, base_img[None, ...].astype(np.int16) + offspring_noises], axis=0)
            population_images = np.clip(population_images, 0, 255).astype(np.uint8)

            selected_indices = self.tournament_selection(ce_pool, tournament_size=4)
            population_noise = population_noises[selected_indices]
            adversarial_images = population_images[selected_indices]
            metrics = {
                "ce": ce_pool[selected_indices],
                "diff_explain": diff_pool[selected_indices],
                "l2": l2_pool[selected_indices],
                "pred": pred_pool[selected_indices],
            }
            # print(metrics)

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
            "quality_ce": self.elite_map.quality_ce,
            "l2": self.elite_map.l2,
            "diff_explain": self.elite_map.diff_explain,
            "pred": self.elite_map.pred,
            "perturbations": self.elite_map.perturbations,
        }


