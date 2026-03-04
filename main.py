import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from attack.flow import Attack, EliteMap
from attack.utils import load_model, seed_everything


def _save_heatmap(data: np.ndarray, title: str, out_path: Path, cmap: str = "viridis") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis_data = np.asarray(data, dtype=np.float32)
    if np.isinf(vis_data).any():
        vis_data = vis_data.copy()
        finite_mask = np.isfinite(vis_data)
        fill_value = float(np.nanmax(vis_data[finite_mask])) if finite_mask.any() else 0.0
        vis_data[~finite_mask] = fill_value

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    image = ax.imshow(vis_data, origin="lower", interpolation="nearest", cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("l2 bin")
    ax.set_ylabel("diff_explain bin")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def visualize_results(results: dict, output_dir: Path) -> None:
    vis_dir = output_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    _save_heatmap(results["quality_ce"], "Quality CE (minimize)", vis_dir / "quality_ce_heatmap.png", cmap="viridis_r")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QD ES black-box attack with descriptors (l2, diff_explain) and quality ce (minimize)."
    )
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results/qd_es_attack"))

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--population_size", type=int, default=128)
    parser.add_argument("--epsilon", type=int, default=32)
    parser.add_argument("--mutation_sigma", type=float, default=0.8)

    parser.add_argument("--l2_max", type=float, default=1.0)
    parser.add_argument("--explain_diff_bins", type=int, default=10)
    parser.add_argument("--l2_diff_bins", type=int, default=10)
    parser.add_argument("--explain_diff_max", type=float, default=1.0)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    model = load_model(args.model_name)
    device = next(model.parameters()).device
    image = Image.open(args.image_path).convert("RGB")

    elite_map = EliteMap(
        explain_diff_bins=args.explain_diff_bins,
        l2_diff_bins=args.l2_diff_bins,
        l2_max=args.l2_max,
        explain_diff_max=args.explain_diff_max,
    )

    algorithm = Attack(
        model=model,
        device=device,
        num_iterations=args.num_iterations,
        population_size=args.population_size,
        epsilon=args.epsilon,
        elite_map=elite_map,
        mutation_sigma=args.mutation_sigma,
    )

    results = algorithm.run(image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "qd_map_metrics.npz",
        quality_ce=results["quality_ce"],
        l2=results["l2"],
        diff_explain=results["diff_explain"],
        pred=results["pred"],
        history_best_ce=np.asarray(results["history_best_ce"], dtype=np.float32),
    )

    perturbation_manifest = []
    perturbation_dir = args.output_dir / "perturbations"
    perturbation_dir.mkdir(parents=True, exist_ok=True)
    for (diff_idx, l2_idx), perturbation in sorted(results["perturbations"].items()):
        perturb_path = perturbation_dir / f"perturb_d{diff_idx}_l{l2_idx}.npy"
        np.save(perturb_path, np.asarray(perturbation, dtype=np.int16))
        perturbation_manifest.append(
            {
                "cell": [int(diff_idx), int(l2_idx)],
                "path": str(perturb_path),
            }
        )

    summary = {
        "model_name": args.model_name,
        "image_path": str(args.image_path),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "num_iterations": args.num_iterations,
        "population_size": args.population_size,
        "epsilon": args.epsilon,
        "mutation_sigma": args.mutation_sigma,
        "descriptors": ["l2", "diff_explain"],
        "quality": "ce",
        "quality_objective": "minimize",
        "original_label": int(results["original_label"]),
        "best_quality_ce": float(results["best_quality_ce"]),
        "occupied_cells": int(results["occupied_cells"]),
        "perturbation_files": perturbation_manifest,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    visualize_results(results, args.output_dir)

    print("\nAttack summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_args())


