import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
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


def class_name(idx: int) -> str:
    if idx < 0:
        return "UNKNOWN"
    if idx < len(IMAGENET_CATEGORIES):
        return IMAGENET_CATEGORIES[idx]
    return f"class_{idx}"


def read_summary(summary_path: Path) -> Dict[str, object]:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_elite_map(margin_loss: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)

    values = np.where(np.isfinite(margin_loss), margin_loss, np.nan)
    image = ax.imshow(values, origin="lower", cmap="viridis", interpolation="nearest")
    ax.set_title("Elite Map (one elite candidate per occupied cell)")
    ax.set_xlabel("L2 bin")
    ax.set_ylabel("IoU bin")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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


def to_model_input(x_rgb: torch.Tensor) -> torch.Tensor:
    return torch.cat([x_rgb, 1.0 - x_rgb], dim=1)


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


def visualize(summary_path: Path, output_dir: Path, max_cells: int, device_arg: str) -> None:
    summary = read_summary(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(summary["model_name"])
    image_path = Path(str(summary["image_path"]))
    original_class = int(summary["original_class"])

    device = select_device(device_arg)
    model = load_model(model_name, device)
    base_x = preprocess_image(model, image_path, device)

    metrics_path = summary_path.parent / "qd_map_metrics.npz"
    metrics = np.load(metrics_path)
    margin_loss = metrics["margin_loss"]
    save_elite_map(margin_loss=margin_loss, out_path=output_dir / "elite_map.png")

    qd_cells: List[Dict[str, object]] = summary.get("qd_cells", [])
    qd_cells = sorted(qd_cells, key=lambda x: float(x["margin_loss"]))
    if max_cells > 0:
        qd_cells = qd_cells[:max_cells]

    cells_dir = output_dir / "candidates"
    cells_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "qd_cells_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "iou_bin",
                "l2_bin",
                "margin_loss",
                "iou",
                "l2",
                "pred_class",
                "pred_name",
                "success",
                "delta_path",
                "attack_image_path",
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

            x_candidate_rgb = reconstruct_candidate_rgb(base_x, delta)
            stem = f"candidate_{index:04d}_iou{iou_bin}_l2{l2_bin}"
            candidate_path = cells_dir / f"{stem}.png"

            pred_class = int(row["pred"])
            pred_name = class_name(pred_class)
            success = bool(row["success"])
            margin_value = float(row["margin_loss"])

            save_candidate_rgb(
                x_candidate_rgb,
                candidate_path,
                title=f"idx={index} margin={margin_value:.4f} pred={pred_class}",
            )

            writer.writerow(
                {
                    "index": index,
                    "iou_bin": iou_bin,
                    "l2_bin": l2_bin,
                    "margin_loss": margin_value,
                    "iou": float(row["iou"]),
                    "l2": float(row["l2"]),
                    "pred_class": pred_class,
                    "pred_name": pred_name,
                    "success": success,
                    "delta_path": str(delta_path),
                    "attack_image_path": str(candidate_path),
                }
            )

    report_summary = {
        "model": model_name,
        "image": str(image_path),
        "original_class": original_class,
        "original_class_name": class_name(original_class),
        "occupied_cells": int(summary.get("occupied_cells", 0)),
        "success_cells": int(summary.get("success_cells", 0)),
        "visualized_cells": len(qd_cells),
        "report_csv": str(csv_path),
        "elite_map": str(output_dir / "elite_map.png"),
    }
    with open(output_dir / "visualization_summary.json", "w", encoding="utf-8") as f:
        json.dump(report_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(report_summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize and report QD ES attack results.")
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.json produced by qd_es_blackbox_attack.py",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/qd_attack_visualization"),
        help="Directory to save plots and reports",
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        default=40,
        help="Maximum number of occupied cells to visualize (sorted by smallest margin loss)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for recomputing explanations",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(
        summary_path=args.summary,
        output_dir=args.output_dir,
        max_cells=args.max_cells,
        device_arg=args.device,
    )