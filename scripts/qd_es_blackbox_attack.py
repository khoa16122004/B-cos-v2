import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
try:
    import bcos.models.pretrained as pretrained
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    import bcos.models.pretrained as pretrained


@dataclass
class AttackConfig:
    model_name: str
    image_path: Path
    output_dir: Path
    iterations: int
    parents: int
    children_per_parent: int
    population_size: int
    l2_max: float
    mutation_sigma_min: float
    mutation_sigma_max: float
    iou_bins: int
    l2_bins: int
    map_threshold_percentile: float
    seed: int
    device: str


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def select_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(model_name: str, device: torch.device) -> torch.nn.Module:
    model = getattr(pretrained, model_name)(pretrained=True)
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def preprocess_image(model: torch.nn.Module, image_path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if hasattr(model.transform, "resize") and hasattr(model.transform, "center_crop"):
        image = model.transform.resize(image)
        image = model.transform.center_crop(image)
    rgb = TF.pil_to_tensor(image).float() / 255.0
    return rgb.unsqueeze(0).to(device)


def to_model_input(x_rgb: torch.Tensor) -> torch.Tensor:
    return torch.cat([x_rgb, 1.0 - x_rgb], dim=1)


def compute_margin_loss(logits: torch.Tensor, target_class: int) -> torch.Tensor:
    target_logits = logits[:, target_class]
    masked = logits.clone()
    masked[:, target_class] = float("-inf")
    best_other_logits = masked.max(dim=1).values
    return target_logits - best_other_logits


def map_to_binary(weight_map: torch.Tensor, percentile: float) -> torch.Tensor:
    positive = weight_map.clamp(min=0)
    flat = positive.flatten()
    if torch.count_nonzero(flat) == 0:
        return torch.zeros_like(positive, dtype=torch.bool)
    threshold = torch.quantile(flat, percentile / 100.0) # take at percentile
    return positive >= threshold


def get_weight_maps_batch(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    x_local = x_batch.detach().clone().requires_grad_(True)
    with torch.enable_grad(), model.explanation_mode():
        out = model(x_local)
        out[:, target_class].sum().backward(inputs=[x_local])
    return (x_local * x_local.grad).sum(1).detach()


def iou_score(binary_a: torch.Tensor, binary_b: torch.Tensor) -> float:
    intersection = torch.logical_and(binary_a, binary_b).sum().item()
    union = torch.logical_or(binary_a, binary_b).sum().item()
    if union == 0:
        return 1.0
    return float(intersection / union)


def l2_norm(delta: torch.Tensor) -> float:
    return float(delta.flatten().norm(p=2).item())


def project_l2(delta: torch.Tensor, l2_max: float) -> torch.Tensor:
    norm = delta.flatten().norm(p=2)
    if norm <= l2_max:
        return delta
    return delta * (l2_max / (norm + 1e-12))


def set_l2(delta: torch.Tensor, target_l2: float) -> torch.Tensor:
    if target_l2 <= 0:
        return torch.zeros_like(delta)

    norm = float(delta.flatten().norm(p=2).item())
    if norm < 1e-12:
        direction = torch.randn_like(delta)
        direction_norm = direction.flatten().norm(p=2)
        return direction * (target_l2 / (direction_norm + 1e-12))

    return delta * (target_l2 / (norm + 1e-12))


class QDMap:
    def __init__(self, iou_bins: int, l2_bins: int, l2_max: float):
        self.iou_bins = iou_bins
        self.l2_bins = l2_bins
        self.l2_max = l2_max

        self.margin_loss = torch.full((iou_bins, l2_bins), float("inf"), dtype=torch.float32)
        self.iou = torch.zeros((iou_bins, l2_bins), dtype=torch.float32)
        self.l2 = torch.zeros((iou_bins, l2_bins), dtype=torch.float32)
        self.pred = torch.full((iou_bins, l2_bins), -1, dtype=torch.long)
        self.success = torch.zeros((iou_bins, l2_bins), dtype=torch.bool)
        self.delta = {}

    def _bin_indices(self, iou_value: float, l2_value: float) -> Tuple[int, int]:
        iou_idx = min(self.iou_bins - 1, max(0, int(iou_value * self.iou_bins)))
        l2_ratio = 0.0 if self.l2_max <= 0 else (l2_value / self.l2_max)
        l2_idx = min(self.l2_bins - 1, max(0, int(l2_ratio * self.l2_bins)))
        return iou_idx, l2_idx

    def try_insert(
        self,
        margin_value: float,
        iou_value: float,
        l2_value: float,
        pred_class: int,
        is_success: bool,
        delta: torch.Tensor,
    ) -> bool:
        i_idx, l_idx = self._bin_indices(iou_value, l2_value)
        if margin_value >= float(self.margin_loss[i_idx, l_idx]):
            return False
        self.margin_loss[i_idx, l_idx] = margin_value
        self.iou[i_idx, l_idx] = iou_value
        self.l2[i_idx, l_idx] = l2_value
        self.pred[i_idx, l_idx] = pred_class
        self.success[i_idx, l_idx] = is_success
        self.delta[(i_idx, l_idx)] = delta.detach().cpu().clone()
        return True

    def occupied_cells(self) -> int:
        return int(torch.isfinite(self.margin_loss).sum().item())

    def elites(self) -> List[Tuple[Tuple[int, int], torch.Tensor, float]]:
        result = []
        for key, value in self.delta.items():
            i_idx, l_idx = key
            result.append((key, value, float(self.margin_loss[i_idx, l_idx].item())))
        result.sort(key=lambda item: item[2])
        return result


def evaluate_candidates_batch(
    model: torch.nn.Module,
    base_rgb: torch.Tensor,
    deltas: torch.Tensor,
    original_class: int,
    original_binary_map: torch.Tensor,
    map_threshold_percentile: float,
) -> List[Dict[str, object]]:
    candidates_rgb = (base_rgb + deltas).clamp(0.0, 1.0)
    candidates_x = to_model_input(candidates_rgb)

    with torch.no_grad():
        logits = model(candidates_x)
        margin_values = compute_margin_loss(logits, original_class)
        pred_classes = logits.argmax(dim=1)

    candidate_weight_maps = get_weight_maps_batch(model, candidates_x, original_class)

    results: List[Dict[str, object]] = []
    for i in range(candidates_x.shape[0]):
        candidate_weight_map = candidate_weight_maps[i]
        candidate_binary = map_to_binary(candidate_weight_map, percentile=map_threshold_percentile)
        iou_value = iou_score(original_binary_map, candidate_binary)
        l2_value = l2_norm(deltas[i : i + 1])
        pred_class = int(pred_classes[i].item())
        is_success = pred_class != original_class

        results.append(
            {
                "margin_loss": float(margin_values[i].item()),
                "pred": pred_class,
                "iou": iou_value,
                "l2": l2_value,
                "success": is_success,
                "delta": deltas[i : i + 1],
            }
        )
    return results


def mutate_from_parent(parent_delta: torch.Tensor, sigma: float, l2_max: float) -> torch.Tensor:
    noise = torch.randn_like(parent_delta) * sigma
    child = parent_delta + noise
    return project_l2(child, l2_max)


def crossover(parent_a: torch.Tensor, parent_b: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * parent_a + (1.0 - alpha) * parent_b


def init_population(population_size: int, sample_shape: Tuple[int, ...], l2_max: float, device: torch.device) -> torch.Tensor:
    if population_size <= 0:
        raise ValueError("population_size must be > 0")

    population = torch.randn((population_size, *sample_shape), device=device)
    flat = population.flatten(1)
    flat_norm = flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
    directions = flat / flat_norm

    l2_targets = torch.linspace(
        max(1e-6, 0.05 * l2_max),
        l2_max,
        steps=population_size,
        device=device,
    )
    l2_targets = l2_targets[torch.randperm(population_size, device=device)]
    scaled = directions * l2_targets[:, None]
    return scaled.view(population.shape)


def select_top_population(
    merged_deltas: torch.Tensor,
    merged_results: List[Dict[str, object]],
    keep_n: int,
) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
    sorted_indices = sorted(
        range(len(merged_results)),
        key=lambda i: float(merged_results[i]["margin_loss"]),
    )[:keep_n]

    index_tensor = torch.tensor(sorted_indices, device=merged_deltas.device, dtype=torch.long)
    next_population = merged_deltas.index_select(0, index_tensor)
    next_results = [merged_results[i] for i in sorted_indices]
    return next_population, next_results


def run_attack(config: AttackConfig) -> Dict[str, object]:
    set_seed(config.seed)
    device = select_device(config.device)
    model = load_model(config.model_name, device)
    print("Loading Model: ", config.model_name)
    base_rgb = preprocess_image(model, config.image_path, device)
    print("Preprocessed image shape (RGB):", base_rgb.shape)
    base_x = to_model_input(base_rgb)
    print("Model input shape (B-cos):", base_x.shape)
    with torch.no_grad():
        base_logits = model(base_x)
        print("Base logits:", base_logits.shape)
        original_class = int(base_logits.argmax(dim=1).item())
        print("Original predicted class:", original_class)
        base_margin_loss = float(compute_margin_loss(base_logits, original_class)[0].item())
        print("Original margin loss:", base_margin_loss)
    base_weight_map = get_weight_maps_batch(model, base_x, original_class)[0]
    base_binary_map = map_to_binary(base_weight_map, percentile=config.map_threshold_percentile)
    # config.output_dir.mkdir(parents=True, exist_ok=True)
    # binary_img = base_binary_map.to(dtype=torch.float32).unsqueeze(0)
    # save_image(binary_img, str(config.output_dir / "base_binary_map.png"))
    # raise
    qd_map = QDMap(config.iou_bins, config.l2_bins, config.l2_max)

    population = init_population(
        population_size=config.population_size,
        sample_shape=tuple(base_rgb.shape[1:]),
        l2_max=config.l2_max,
        device=device,
    )

    population_results = evaluate_candidates_batch(
        model,
        base_rgb,
        population,
        original_class,
        base_binary_map,
        config.map_threshold_percentile,
    )
    for result in population_results:
        qd_map.try_insert(
            margin_value=result["margin_loss"],
            iou_value=result["iou"],
            l2_value=result["l2"],
            pred_class=result["pred"],
            is_success=result["success"],
            delta=result["delta"],
        )

    history_best_margin_loss = [
        float(min(float(result["margin_loss"]) for result in population_results))
    ]
    history_success_cells = [int(qd_map.success.sum().item())]

    for step in range(config.iterations):
        num_candidates = population.shape[0]

        children = []
        for _ in range(num_candidates):
            parent_a_idx = int(np.random.randint(num_candidates))
            parent_b_idx = int(np.random.randint(num_candidates))
            parent_a = population[parent_a_idx : parent_a_idx + 1]
            parent_b = population[parent_b_idx : parent_b_idx + 1]

            alpha = float(np.random.uniform(0.2, 0.8))
            child_delta = crossover(parent_a, parent_b, alpha=alpha)
            child_delta = project_l2(child_delta, l2_max=config.l2_max)
            children.append(child_delta)

        children = torch.cat(children, dim=0)

        children_results = evaluate_candidates_batch(
            model,
            base_rgb,
            children,
            original_class,
            base_binary_map,
            config.map_threshold_percentile,
        )

        for result in children_results:
            qd_map.try_insert(
                margin_value=result["margin_loss"],
                iou_value=result["iou"],
                l2_value=result["l2"],
                pred_class=result["pred"],
                is_success=result["success"],
                delta=result["delta"],
            )

        merged_population = torch.cat([population, children], dim=0)
        merged_results = population_results + children_results
        population, population_results = select_top_population(
            merged_deltas=merged_population,
            merged_results=merged_results,
            keep_n=config.population_size,
        )

        finite_mask = torch.isfinite(qd_map.margin_loss)
        best_margin_loss = float(qd_map.margin_loss[finite_mask].min().item())
        history_best_margin_loss.append(best_margin_loss)
        history_success_cells.append(int(qd_map.success.sum().item()))

        print(
            f"[Iter {step + 1:03d}] best_margin={best_margin_loss:.6f} occupied={qd_map.occupied_cells()} success_cells={int(qd_map.success.sum().item())} parents={config.population_size} children={num_candidates} eval_batch={num_candidates}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.output_dir / "qd_map_metrics.npz",
        margin_loss=qd_map.margin_loss.numpy(),
        iou=qd_map.iou.numpy(),
        l2=qd_map.l2.numpy(),
        pred=qd_map.pred.numpy(),
        success=qd_map.success.numpy(),
        history_best_margin_loss=np.array(history_best_margin_loss, dtype=np.float32),
        history_success_cells=np.array(history_success_cells, dtype=np.int32),
    )

    all_elites = qd_map.elites()
    all_elites_dir = config.output_dir / "all_elites"
    all_elites_dir.mkdir(parents=True, exist_ok=True)
    qd_cells = []
    for (i_idx, l_idx), delta_tensor, margin_value in all_elites:
        path = all_elites_dir / f"elite_iou{i_idx}_l2{l_idx}.pt"
        torch.save(delta_tensor, path)
        qd_cells.append(
            {
                "cell": [i_idx, l_idx],
                "margin_loss": margin_value,
                "iou": float(qd_map.iou[i_idx, l_idx].item()),
                "l2": float(qd_map.l2[i_idx, l_idx].item()),
                "pred": int(qd_map.pred[i_idx, l_idx].item()),
                "success": bool(qd_map.success[i_idx, l_idx].item()),
                "delta_path": str(path),
            }
        )

    summary = {
        "model_name": config.model_name,
        "image_path": str(config.image_path),
        "device": str(device),
        "original_class": original_class,
        "base_margin_loss": base_margin_loss,
        "iterations": config.iterations,
        "occupied_cells": qd_map.occupied_cells(),
        "success_cells": int(qd_map.success.sum().item()),
        "best_margin_loss": float(min(history_best_margin_loss)),
        "history_best_margin_loss": history_best_margin_loss,
        "qd_cells": qd_cells,
        "top_elites": qd_cells[:10],
    }
    with open(config.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def parse_args() -> AttackConfig:
    parser = argparse.ArgumentParser(
        description="Black-box Evolution Strategy QD attack with descriptors (IoU weight-map, L2)."
    )
    parser.add_argument("--model", default="resnet50", help="Model entrypoint from bcos.models.pretrained")
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--output_dir", type=Path, default=Path("results/qd_es_attack"))
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--parents", type=int, default=4)
    parser.add_argument("--children_per_parent", type=int, default=8)
    parser.add_argument("--population_size", type=int, default=256)
    parser.add_argument("--l2_max", type=float, default=22.0)
    parser.add_argument("--mutation_sigma_min", type=float, default=0.05)
    parser.add_argument("--mutation_sigma_max", type=float, default=0.35)
    parser.add_argument("--iou_bins", type=int, default=16)
    parser.add_argument("--l2_bins", type=int, default=16)
    parser.add_argument("--map_threshold_percentile", type=float, default=85.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()
    return AttackConfig(
        model_name=args.model,
        image_path=args.image,
        output_dir=args.output_dir,
        iterations=args.iterations,
        parents=args.parents,
        children_per_parent=args.children_per_parent,
        population_size=args.population_size,
        l2_max=args.l2_max,
        mutation_sigma_min=args.mutation_sigma_min,
        mutation_sigma_max=args.mutation_sigma_max,
        iou_bins=args.iou_bins,
        l2_bins=args.l2_bins,
        map_threshold_percentile=args.map_threshold_percentile,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    attack_config = parse_args()
    result = run_attack(attack_config)
    print("\nAttack summary")
    print(json.dumps({
        "original_class": result["original_class"],
        "base_margin_loss": result["base_margin_loss"],
        "best_margin_loss": result["best_margin_loss"],
        "occupied_cells": result["occupied_cells"],
        "success_cells": result["success_cells"],
    }, indent=2))