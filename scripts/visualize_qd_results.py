import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

try:
    import bcos.models.pretrained as pretrained
    from bcos.data.categories import IMAGENET_CATEGORIES
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    import bcos.models.pretrained as pretrained
    from bcos.data.categories import IMAGENET_CATEGORIES


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


def class_name(idx: int) -> str:
    if idx < 0:
        return "UNKNOWN"
    if idx < len(IMAGENET_CATEGORIES):
        return IMAGENET_CATEGORIES[idx]
    return f"class_{idx}"


def save_candidate_rgb(x_candidate_rgb: torch.Tensor, out_path: Path, title: str) -> None:
    rgb = x_candidate_rgb.detach().cpu()[0].permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    ax.imshow(rgb)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_weight_map(weight_map: torch.Tensor, out_path: Path, title: str) -> None:
    arr = weight_map.detach().cpu().numpy()
    vmax = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    image = ax.imshow(arr, cmap="bwr", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_elite_map(quality_map: np.ndarray, out_path: Path, l2_max: float, quality_label: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    values = np.where(np.isfinite(quality_map), quality_map, np.nan)
    image = ax.imshow(
        values,
        origin="lower",
        cmap="viridis_r",
        interpolation="nearest",
        extent=(0.0, float(l2_max), 0.0, 1.0),
        aspect="auto",
    )
    ax.set_title(f"Elite map ({quality_label}, lower is better)")
    ax.set_xlabel("L2")
    ax.set_ylabel("IoU")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def reconstruct_candidate_rgb(base_rgb: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    if delta.ndim == 3:
        delta = delta.unsqueeze(0)

    if delta.shape[1] == 3:
        return (base_rgb + delta).clamp(0.0, 1.0)

    if delta.shape[1] == 6:
        base_x = to_model_input(base_rgb)
        x_adv = (base_x + delta).clamp(0.0, 1.0)
        return x_adv[:, :3]

    raise ValueError(f"Unsupported delta channel size: {delta.shape[1]}")


def get_prediction_info(model: torch.nn.Module, x_candidate: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        logits = model(x_candidate)
        pred_idx = int(logits.argmax(dim=1).item())
        if hasattr(model, "to_probabilities"):
            probs = model.to_probabilities(logits)
        else:
            probs = F.softmax(logits, dim=1)
        pred_score = float(probs[0, pred_idx].item())
    return {"pred_class": pred_idx, "pred_score": pred_score}


def visualize(summary_path: Path, output_dir: Path, max_cells: int, device_arg: str) -> None:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(summary["model_name"])
    image_path = Path(str(summary["image_path"]))
    original_class = int(summary["original_class"])

    device = select_device(device_arg)
    model = load_model(model_name, device)
    base_rgb = preprocess_image(model, image_path, device)

    metrics_path = summary_path.parent / "qd_map_metrics.npz"
    metrics = np.load(metrics_path)
    if "quality_ce" in metrics:
        quality_key = "quality_ce"
    elif "quality_l2" in metrics:
        quality_key = "quality_l2"
    elif "margin_loss" in metrics:
        quality_key = "margin_loss"
    else:
        raise KeyError("Cannot find quality map in metrics file.")

    quality_map = metrics[quality_key]
    l2_max = float(summary.get("l2_max", np.nanmax(metrics["l2"]) if "l2" in metrics else 1.0))
    elite_map_path = output_dir / "elite_map.png"
    save_elite_map(quality_map=quality_map, out_path=elite_map_path, l2_max=l2_max, quality_label=quality_key)

    qd_cells: List[Dict[str, object]] = summary.get("qd_cells", [])
    def row_quality(row: Dict[str, object]) -> float:
        if "quality_ce" in row:
            return float(row["quality_ce"])
        if "quality_l2" in row:
            return float(row["quality_l2"])
        if "margin_loss" in row:
            return float(row["margin_loss"])
        return float("inf")

    qd_cells = sorted(qd_cells, key=row_quality)
    if max_cells > 0:
        qd_cells = qd_cells[:max_cells]

    cells_dir = output_dir / "candidates"
    cells_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "qd_cells_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "index",
                "iou_bin",
                "l2_bin",
                "quality",
                "quality_name",
                "margin_loss",
                "iou",
                "l2",
                "pred_class",
                "pred_name",
                "pred_score",
                "consistent",
                "delta_path",
                "candidate_image_path",
                "weight_map_path",
            ],
        )
        writer.writeheader()

        for index, row in enumerate(qd_cells):
            iou_bin = int(row["cell"][0])
            l2_bin = int(row["cell"][1])
            delta_path = Path(str(row["delta_path"]))
            if not delta_path.exists():
                continue

            delta = torch.load(delta_path, map_location=device)
            if delta.ndim == 3:
                delta = delta.unsqueeze(0)

            x_candidate_rgb = reconstruct_candidate_rgb(base_rgb, delta)
            x_candidate = to_model_input(x_candidate_rgb)

            stem = f"candidate_{index:04d}_iou{iou_bin}_l2{l2_bin}"
            candidate_path = cells_dir / f"{stem}.png"
            weight_map_path = cells_dir / f"{stem}_weight_map.png"

            pred_info = get_prediction_info(model, x_candidate)
            pred_class = int(pred_info["pred_class"])
            pred_score = float(pred_info["pred_score"])
            pred_name = class_name(pred_class)

            x_local = x_candidate.detach().clone().requires_grad_(True)
            explanation = model.explain(x_local, idx=original_class)
            weight_map = explanation["contribution_map"][0].detach()

            margin_value = float(row.get("margin_loss", np.nan))
            quality_value = row_quality(row)
            save_candidate_rgb(
                x_candidate_rgb,
                candidate_path,
                title=f"idx={index} {quality_key}={quality_value:.4f} pred={pred_class} score={pred_score:.4f}",
            )
            save_weight_map(
                weight_map,
                weight_map_path,
                title=f"Weight map idx={index}",
            )

            writer.writerow(
                {
                    "index": index,
                    "iou_bin": iou_bin,
                    "l2_bin": l2_bin,
                    "quality": quality_value,
                    "quality_name": quality_key,
                    "margin_loss": margin_value,
                    "iou": float(row.get("iou", np.nan)),
                    "l2": float(row.get("l2", np.nan)),
                    "pred_class": pred_class,
                    "pred_name": pred_name,
                    "pred_score": pred_score,
                    "consistent": bool(row.get("consistent", not bool(row.get("success", False)))),
                    "delta_path": str(delta_path),
                    "candidate_image_path": str(candidate_path),
                    "weight_map_path": str(weight_map_path),
                }
            )

    report_summary = {
        "model": model_name,
        "image": str(image_path),
        "original_class": original_class,
        "occupied_cells": int(summary.get("occupied_cells", 0)),
        "consistent_cells": int(summary.get("consistent_cells", summary.get("success_cells", 0))),
        "quality_name": quality_key,
        "visualized_cells": len(qd_cells),
        "elite_map": str(elite_map_path),
        "report_csv": str(csv_path),
    }
    with open(output_dir / "visualization_summary.json", "w", encoding="utf-8") as f:
        json.dump(report_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(report_summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize QD optimization results")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary.json")
    parser.add_argument("--output_dir", type=Path, default=Path("results/qd_visualization"))
    parser.add_argument("--max_cells", type=int, default=40)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(
        summary_path=args.summary,
        output_dir=args.output_dir,
        max_cells=args.max_cells,
        device_arg=args.device,
    )
