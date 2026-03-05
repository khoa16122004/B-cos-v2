import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from attack.nsga import NSGABcosAttack
from attack.utils import load_model, save_map_image, seed_everything


def _to_numpy(arr_like) -> np.ndarray:
    if isinstance(arr_like, torch.Tensor):
        arr = arr_like.detach().cpu().numpy()
    else:
        arr = np.asarray(arr_like)
    return np.asarray(arr)


def _to_hwc3(arr_like) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D explanation map, got shape={arr.shape}")
    if arr.shape[0] <= 4 and arr.shape[-1] > 4:
        arr = np.transpose(arr, (1, 2, 0))
    return np.asarray(arr)


def _to_2d(arr_like) -> np.ndarray:
    arr = np.asarray(arr_like)
    if arr.ndim >= 3:
        arr = arr[0]
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D contribution map, got shape={arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def _save_contribution_heatmap(map_2d: np.ndarray, output_path: Path) -> None:
    vmax = float(np.percentile(np.abs(map_2d), 99.0))
    if not np.isfinite(vmax) or vmax <= 1e-12:
        vmax = 1.0

    plt.figure(figsize=(4, 4), dpi=140)
    plt.imshow(map_2d, cmap="bwr", vmin=-vmax, vmax=vmax)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NSGA-II black-box attack for B-cos models with two objectives: "
            "minimize CE(gt) and optimize evaluate-score."
        )
    )
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("results/nsga_attack"))

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--population_size", type=int, default=50)
    parser.add_argument("--epsilon", type=int, default=32)
    parser.add_argument("--mutation_sigma", type=float, default=0.35)
    parser.add_argument("--crossover_alpha_min", type=float, default=0.2)
    parser.add_argument("--crossover_alpha_max", type=float, default=0.8)

    parser.add_argument(
        "--score_mode",
        type=str,
        default="mean",
        choices=["descriptor_1", "descriptor_2", "mean"],
        help="Score used as objective #2 from evaluate().",
    )
    parser.add_argument(
        "--score_objective",
        type=str,
        default="min",
        choices=["min", "max"],
        help="Direction for objective #2.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    model = load_model(args.model_name)
    device = next(model.parameters()).device
    image = Image.open(args.image_path).convert("RGB")

    attack = NSGABcosAttack(
        model=model,
        device=device,
        num_iterations=args.num_iterations,
        population_size=args.population_size,
        epsilon=args.epsilon,
        mutation_sigma=args.mutation_sigma,
        crossover_alpha_min=args.crossover_alpha_min,
        crossover_alpha_max=args.crossover_alpha_max,
        score_mode=args.score_mode,
        score_objective=args.score_objective,
    )

    results = attack.run(image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "nsga_metrics.npz",
        history_best_ce=np.asarray(results.history_best_ce, dtype=np.float32),
        history_best_score=np.asarray(results.history_best_score, dtype=np.float32),
        pareto_ce=np.asarray(results.pareto_ce, dtype=np.float32),
        pareto_score=np.asarray(results.pareto_score, dtype=np.float32),
        pareto_pred=np.asarray(results.pareto_pred, dtype=np.int64),
    )

    np.save(args.output_dir / "history_best_ce.npy", np.asarray(results.history_best_ce, dtype=np.float32))
    np.save(args.output_dir / "history_best_score.npy", np.asarray(results.history_best_score, dtype=np.float32))

    archive_history_payload = {
        "archive_ce_history": [np.asarray(x, dtype=np.float32) for x in results.archive_ce_history],
        "archive_score_history": [np.asarray(x, dtype=np.float32) for x in results.archive_score_history],
        "archive_pred_history": [np.asarray(x, dtype=np.int64) for x in results.archive_pred_history],
    }
    with open(args.output_dir / "archive_history.pkl", "wb") as f:
        pickle.dump(archive_history_payload, f)

    np.save(args.output_dir / "best_perturbation.npy", np.asarray(results.best_perturbation, dtype=np.int16))
    best_adv_path = args.output_dir / "best_adversarial.png"
    Image.fromarray(np.asarray(results.best_adversarial, dtype=np.uint8)).save(best_adv_path)

    clean_img = attack.spatial_transform(image)
    clean_batch = np.expand_dims(np.asarray(clean_img, dtype=np.uint8), axis=0)
    clean_tensor = attack._to_model_batch(clean_batch).requires_grad_(True)
    clean_explain = model.explain(clean_tensor, int(results.original_label))

    best_adv_batch = np.expand_dims(np.asarray(results.best_adversarial, dtype=np.uint8), axis=0)
    best_tensor = attack._to_model_batch(best_adv_batch).requires_grad_(True)
    best_explain = model.explain(best_tensor, int(results.original_label))

    original_explanation = _to_hwc3(_to_numpy(clean_explain["explanation"]))
    best_explanation = _to_hwc3(_to_numpy(best_explain["explanation"]))

    original_contribution = _to_2d(_to_numpy(clean_explain["contribution_map"]))
    best_contribution = _to_2d(_to_numpy(best_explain["contribution_map"]))

    best_explanation_npy = args.output_dir / "best_explanation_map.npy"
    best_explanation_png = args.output_dir / "best_explanation_map.png"
    best_contribution_npy = args.output_dir / "best_contribution_map.npy"
    best_contribution_png = args.output_dir / "best_contribution_map.png"
    original_explanation_npy = args.output_dir / "original_explanation_map.npy"
    original_explanation_png = args.output_dir / "original_explanation_map.png"
    original_contribution_npy = args.output_dir / "original_contribution_map.npy"
    original_contribution_png = args.output_dir / "original_contribution_map.png"

    np.save(original_explanation_npy, original_explanation)
    np.save(original_contribution_npy, original_contribution)
    np.save(best_explanation_npy, best_explanation)
    np.save(best_contribution_npy, best_contribution)

    save_map_image(original_explanation, original_explanation_png)
    save_map_image(best_explanation, best_explanation_png)
    _save_contribution_heatmap(original_contribution, original_contribution_png)
    _save_contribution_heatmap(best_contribution, best_contribution_png)

    np.save(
        args.output_dir / "final_archive_perturbations.npy",
        np.asarray(results.final_archive_perturbations, dtype=np.int16),
    )
    np.save(
        args.output_dir / "final_archive_adversarials.npy",
        np.asarray(results.final_archive_adversarials, dtype=np.uint8),
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
        "crossover_alpha_min": args.crossover_alpha_min,
        "crossover_alpha_max": args.crossover_alpha_max,
        "score_mode": args.score_mode,
        "score_objective": args.score_objective,
        "original_label": int(results.original_label),
        "best_ce": float(results.best_ce),
        "best_score": float(results.best_score),
        "class_preserved": bool(results.class_preserved),
        "pareto_count": int(len(results.pareto_ce)),
        "archive_iterations": int(len(results.archive_score_history)),
        "best_adversarial_path": str(best_adv_path),
        "best_explanation_map_npy": str(best_explanation_npy),
        "best_explanation_map_png": str(best_explanation_png),
        "best_contribution_map_npy": str(best_contribution_npy),
        "best_contribution_map_png": str(best_contribution_png),
        "original_explanation_map_npy": str(original_explanation_npy),
        "original_explanation_map_png": str(original_explanation_png),
        "original_contribution_map_npy": str(original_contribution_npy),
        "original_contribution_map_png": str(original_contribution_png),
    }

    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nNSGA-II attack summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_args())
