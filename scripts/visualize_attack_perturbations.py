import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

try:
    from attack.utils import load_model, save_map_image
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from attack.utils import load_model, save_map_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize explanations from saved per-cell perturbations."
    )
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary.json from main.py")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for visualizations (default: <summary_dir>/perturbation_visualization)",
    )
    return parser.parse_args()


def load_summary(summary_path: Path) -> dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_numpy_map(explanation_map) -> np.ndarray:
    if isinstance(explanation_map, torch.Tensor):
        arr = explanation_map.detach().cpu().numpy()
    else:
        arr = np.asarray(explanation_map)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected explanation map to have 3 dims, got shape={arr.shape}")
    return arr


def split_attack_transforms(model) -> tuple:
    transform_obj = model.transform
    if hasattr(transform_obj, "transforms") and hasattr(transform_obj.transforms, "transforms"):
        transform_list = list(transform_obj.transforms.transforms)
    elif hasattr(transform_obj, "transforms"):
        transform_list = list(transform_obj.transforms)
    else:
        raise ValueError("Cannot split model.transform into spatial and bcos parts.")

    if len(transform_list) < 3:
        raise ValueError("model.transform has fewer than 3 steps; cannot split with current attack rule.")

    spatial_transform = transforms.Compose(transform_list[:-3])
    bcos_transform = transforms.Compose(transform_list[-3:])
    return spatial_transform, bcos_transform


def main() -> None:
    args = parse_args()
    summary = load_summary(args.summary)

    if "perturbation_files" not in summary or len(summary["perturbation_files"]) == 0:
        raise ValueError("summary.json does not contain 'perturbation_files'. Run main.py again to generate them.")

    model_name = str(summary["model_name"])
    image_path = Path(str(summary["image_path"]))
    output_dir = args.output_dir or (args.summary.parent / "perturbation_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(model_name)
    device = next(model.parameters()).device
    spatial_transform, bcos_transform = split_attack_transforms(model)

    base_pil = Image.open(image_path).convert("RGB")
    spatial_img = spatial_transform(base_pil)
    base_img = np.asarray(spatial_img, dtype=np.uint8).copy()

    report = []
    for row in tqdm(summary["perturbation_files"], desc="Visualizing perturbation cells"):
        cell = row["cell"]
        descriptor_1_idx, descriptor_2_idx = int(cell[0]), int(cell[1])
        perturb_path = Path(str(row["path"]))
        if not perturb_path.exists():
            continue

        perturbation = np.load(perturb_path)
        adv_img = np.clip(base_img.astype(np.int16) + perturbation.astype(np.int16), 0, 255).astype(np.uint8).copy()

        adv_pil = Image.fromarray(adv_img)
        adv_tensor = bcos_transform(adv_pil).unsqueeze(0).to(device).requires_grad_(True)
        explain_result = model.explain(adv_tensor)
        explanation_map = to_numpy_map(explain_result["explanation"])

        adv_path = output_dir / f"cell_d{descriptor_1_idx}_l{descriptor_2_idx}_adv.png"
        exp_path = output_dir / f"cell_d{descriptor_1_idx}_l{descriptor_2_idx}_explanation.png"
        Image.fromarray(adv_img).save(adv_path)
        save_map_image(explanation_map, exp_path)

        report.append(
            {
                "cell": [descriptor_1_idx, descriptor_2_idx],
                "perturbation_path": str(perturb_path),
                "adversarial_image_path": str(adv_path),
                "explanation_path": str(exp_path),
            }
        )

    with open(output_dir / "visualization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(report)} perturbation visualizations to {output_dir}")


if __name__ == "__main__":
    main()
