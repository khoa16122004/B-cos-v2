import argparse
import csv
import json
import pickle
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from attack.utils import load_model
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from attack.utils import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize NSGA Pareto front per iteration from archive_history.pkl. "
            "Objective 1 is CE (lower is better), objective 2 is score (depends on score_objective)."
        )
    )
    parser.add_argument(
        "--sample_dir",
        type=Path,
        required=True,
        help=(
            "Folder of one sample run (contains archive_history.pkl, summary.json, "
            "final_archive_adversarials.npy)."
        ),
    )
    parser.add_argument(
        "--archive_history",
        type=Path,
        default=None,
        help="Optional override path to archive_history.pkl (default: <sample_dir>/archive_history.pkl)",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional override path to summary.json (default: <sample_dir>/summary.json if exists).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory (default: <archive_dir>/pareto_viz)",
    )
    parser.add_argument(
        "--no_gif",
        action="store_true",
        help="Disable GIF export (by default script attempts to save GIF).",
    )
    parser.add_argument(
        "--final_archive_adversarials",
        type=Path,
        default=None,
        help="Optional override (default: <sample_dir>/final_archive_adversarials.npy)",
    )
    parser.add_argument(
        "--export_final_maps",
        action="store_true",
        help="Export adversarial image + explanation map + contribution map for each final Pareto member.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for map export. If omitted, read from summary.json.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when running model.explain for final archive export.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second for GIF export.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="DPI for saved figures.",
    )
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Optional[Path], Optional[Path], Path]:
    sample_dir = args.sample_dir
    if not sample_dir.exists() or not sample_dir.is_dir():
        raise FileNotFoundError(f"sample_dir is not a valid folder: {sample_dir}")

    archive_path = args.archive_history or (sample_dir / "archive_history.pkl")
    if not archive_path.exists():
        raise FileNotFoundError(f"archive_history not found: {archive_path}")

    summary_path = args.summary
    if summary_path is None:
        default_summary = sample_dir / "summary.json"
        if default_summary.exists():
            summary_path = default_summary

    final_archive_path = args.final_archive_adversarials
    if final_archive_path is None:
        default_final = sample_dir / "final_archive_adversarials.npy"
        if default_final.exists():
            final_archive_path = default_final

    output_dir = args.output_dir or (sample_dir / "pareto_viz")
    return archive_path, summary_path, final_archive_path, output_dir


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}, got {type(payload)}")
    return payload


def load_summary(path: Optional[Path]) -> dict:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in {path}, got {type(payload)}")
    return payload


def get_axis_limits(ce_history: List[np.ndarray], score_history: List[np.ndarray]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    all_ce = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in ce_history if np.asarray(x).size > 0], axis=0)
    all_score = np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in score_history if np.asarray(x).size > 0], axis=0)

    ce_min, ce_max = float(np.min(all_ce)), float(np.max(all_ce))
    score_min, score_max = float(np.min(all_score)), float(np.max(all_score))

    ce_pad = max(1e-6, 0.05 * (ce_max - ce_min + 1e-6))
    score_pad = max(1e-6, 0.05 * (score_max - score_min + 1e-6))

    return (ce_min - ce_pad, ce_max + ce_pad), (score_min - score_pad, score_max + score_pad)


def ensure_1d_list(arr_list: List[np.ndarray], name: str) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for idx, arr in enumerate(arr_list):
        np_arr = np.asarray(arr)
        if np_arr.ndim == 0:
            np_arr = np_arr.reshape(1)
        out.append(np_arr.reshape(-1))
        if out[-1].dtype.kind not in "fiu":
            raise ValueError(f"{name}[{idx}] must be numeric array")
    return out


def save_frames(
    ce_history: List[np.ndarray],
    score_history: List[np.ndarray],
    pred_history: Optional[List[np.ndarray]],
    output_dir: Path,
    original_label: Optional[int],
    score_objective: str,
    dpi: int,
) -> List[Path]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    xlim, ylim = get_axis_limits(ce_history, score_history)

    frame_paths: List[Path] = []
    n_iters = len(ce_history)

    for i in range(n_iters):
        ce = ce_history[i]
        score = score_history[i]

        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

        if pred_history is not None and original_label is not None and i < len(pred_history):
            pred = pred_history[i]
            if pred.shape[0] == ce.shape[0]:
                same = pred == int(original_label)
                diff = ~same
                if np.any(same):
                    ax.scatter(ce[same], score[same], s=28, c="#1f77b4", alpha=0.9, label="pred == gt")
                if np.any(diff):
                    ax.scatter(ce[diff], score[diff], s=28, c="#d62728", alpha=0.9, label="pred != gt")
                ax.legend(loc="best")
            else:
                ax.scatter(ce, score, s=30, c="#1f77b4", alpha=0.9)
        else:
            ax.scatter(ce, score, s=30, c="#1f77b4", alpha=0.9)

        best_idx = int(np.argmin(ce))
        ax.scatter([ce[best_idx]], [score[best_idx]], s=70, c="#2ca02c", marker="*", label="min CE")

        title = f"Pareto Archive - Iteration {i}"
        ax.set_title(title)
        ax.set_xlabel("CE(gt) [minimize]")
        if score_objective == "min":
            ax.set_ylabel("Score [minimize]")
        else:
            ax.set_ylabel("Score [maximize]")

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if pred_history is None or original_label is None:
            ax.legend(loc="best")

        frame_path = frames_dir / f"pareto_iter_{i:04d}.png"
        fig.tight_layout()
        fig.savefig(frame_path)
        plt.close(fig)
        frame_paths.append(frame_path)

    return frame_paths


def export_gif(frame_paths: List[Path], output_path: Path, fps: int) -> bool:
    try:
        import imageio.v2 as imageio
    except Exception:
        return False

    images = [imageio.imread(path) for path in frame_paths]
    duration = 1.0 / max(1, int(fps))
    imageio.mimsave(output_path, images, duration=duration)
    return True


def save_best_score_curve(
    score_history: List[np.ndarray],
    score_objective: str,
    output_path: Path,
    dpi: int,
) -> np.ndarray:
    best_scores: List[float] = []
    for arr in score_history:
        if arr.size == 0:
            best_scores.append(np.nan)
            continue
        if score_objective == "min":
            best_scores.append(float(np.min(arr)))
        else:
            best_scores.append(float(np.max(arr)))

    y = np.asarray(best_scores, dtype=np.float32)
    x = np.arange(y.shape[0], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    ax.plot(x, y, color="#1f77b4", linewidth=2)
    ax.scatter(x, y, color="#1f77b4", s=14)
    ax.set_title("Best Score per Iteration (Pareto Archive)")
    ax.set_xlabel("Iteration")
    if score_objective == "min":
        ax.set_ylabel("Best Score (min)")
    else:
        ax.set_ylabel("Best Score (max)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return y


def save_best_ce_curve(
    ce_history: List[np.ndarray],
    output_path: Path,
    dpi: int,
) -> np.ndarray:
    best_ce: List[float] = []
    for arr in ce_history:
        if arr.size == 0:
            best_ce.append(np.nan)
            continue
        best_ce.append(float(np.min(arr)))

    y = np.asarray(best_ce, dtype=np.float32)
    x = np.arange(y.shape[0], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    ax.plot(x, y, color="#d62728", linewidth=2)
    ax.scatter(x, y, color="#d62728", s=14)
    ax.set_title("Best CE per Iteration (Pareto Archive)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best CE (min)")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return y


def save_iteration_csv(
    out_csv: Path,
    best_ce_curve: np.ndarray,
    best_score_curve: np.ndarray,
) -> None:
    n = int(max(best_ce_curve.shape[0], best_score_curve.shape[0]))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "best_ce", "best_score"])
        for i in range(n):
            ce_v = float(best_ce_curve[i]) if i < best_ce_curve.shape[0] else np.nan
            score_v = float(best_score_curve[i]) if i < best_score_curve.shape[0] else np.nan
            writer.writerow([i, ce_v, score_v])


def _normalize_to_uint8_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    denom = x_max - x_min
    if denom < 1e-12:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - x_min) / denom
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)


def _extract_2d_map(sample: np.ndarray) -> np.ndarray:
    sample = np.asarray(sample)
    if sample.ndim == 2:
        return sample
    if sample.ndim == 3:
        # Handle [C,H,W] or [H,W,C] by averaging channel axis.
        if sample.shape[0] <= 4:
            return np.mean(sample, axis=0)
        if sample.shape[-1] <= 4:
            return np.mean(sample, axis=-1)
    return np.squeeze(sample)


def export_final_archive_maps(
    final_archive_adversarials_path: Path,
    output_dir: Path,
    model_name: str,
    gt_label: int,
    batch_size: int,
) -> dict:
    if not final_archive_adversarials_path.exists():
        raise FileNotFoundError(f"final archive adversarial file not found: {final_archive_adversarials_path}")

    adv_imgs = np.asarray(np.load(final_archive_adversarials_path), dtype=np.uint8)
    if adv_imgs.ndim != 4 or adv_imgs.shape[-1] not in {1, 3, 4}:
        raise ValueError(
            "final_archive_adversarials.npy must be [N,H,W,C] uint8 images, got "
            f"shape={adv_imgs.shape}"
        )

    maps_dir = output_dir / "final_archive_maps"
    adv_dir = maps_dir / "adversarial"
    explain_dir = maps_dir / "explanation"
    contribution_dir = maps_dir / "contribution"
    for p in [maps_dir, adv_dir, explain_dir, contribution_dir]:
        p.mkdir(parents=True, exist_ok=True)

    model = load_model(model_name)
    device = next(model.parameters()).device
    bcos_transform_class = model.transform
    bcos_transform = transforms.Compose(bcos_transform_class.transforms.transforms[-3:])

    n = int(adv_imgs.shape[0])
    preds_all: List[np.ndarray] = []
    explain_all: List[np.ndarray] = []
    contribution_all: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n, max(1, int(batch_size))):
            end = min(n, start + max(1, int(batch_size)))
            imgs_pil = [Image.fromarray(img.astype(np.uint8)) for img in adv_imgs[start:end]]
            batch = torch.stack([bcos_transform(img) for img in imgs_pil], dim=0).to(device)
            out = model.explain(batch, int(gt_label))

            preds = out["prediction"]
            if isinstance(preds, torch.Tensor):
                preds_np = preds.detach().cpu().numpy().reshape(-1)
            else:
                preds_np = np.asarray(preds).reshape(-1)
            preds_all.append(preds_np.astype(np.int64, copy=True))

            explain = out["explanation"]
            if isinstance(explain, torch.Tensor):
                explain_np = explain.detach().cpu().numpy()
            else:
                explain_np = np.asarray(explain)
            explain_all.append(explain_np.copy())

            contribution = out["contribution_map"]
            if isinstance(contribution, torch.Tensor):
                contribution_np = contribution.detach().cpu().numpy()
            else:
                contribution_np = np.asarray(contribution)
            contribution_all.append(contribution_np.copy())

    preds_np = np.concatenate(preds_all, axis=0)
    explain_np = np.concatenate(explain_all, axis=0)
    contribution_np = np.concatenate(contribution_all, axis=0)

    for i in range(n):
        adv = adv_imgs[i]
        if adv.shape[-1] == 1:
            adv_to_save = adv[..., 0]
        elif adv.shape[-1] == 4:
            adv_to_save = adv[..., :3]
        else:
            adv_to_save = adv
        Image.fromarray(adv_to_save.astype(np.uint8)).save(adv_dir / f"adv_{i:04d}.png")

        explain_2d = _extract_2d_map(explain_np[i])
        contribution_2d = _extract_2d_map(contribution_np[i])
        Image.fromarray(_normalize_to_uint8_2d(explain_2d)).save(explain_dir / f"explain_{i:04d}.png")
        Image.fromarray(_normalize_to_uint8_2d(contribution_2d)).save(contribution_dir / f"contribution_{i:04d}.png")

    np.save(maps_dir / "predictions.npy", preds_np.astype(np.int64))
    np.save(maps_dir / "explanation_raw.npy", explain_np)
    np.save(maps_dir / "contribution_raw.npy", contribution_np)

    report = {
        "count": n,
        "gt_label": int(gt_label),
        "same_as_gt_count": int(np.sum(preds_np == int(gt_label))),
        "maps_dir": str(maps_dir),
        "adversarial_dir": str(adv_dir),
        "explanation_dir": str(explain_dir),
        "contribution_dir": str(contribution_dir),
    }
    return report


def main() -> None:
    args = parse_args()
    archive_path, summary_path, final_archive_default_path, out_dir = resolve_inputs(args)

    summary = load_summary(summary_path)

    payload = load_pickle(archive_path)
    if "archive_ce_history" not in payload or "archive_score_history" not in payload:
        raise KeyError("archive_history.pkl must contain keys: archive_ce_history, archive_score_history")

    ce_history = ensure_1d_list(payload["archive_ce_history"], "archive_ce_history")
    score_history = ensure_1d_list(payload["archive_score_history"], "archive_score_history")

    if len(ce_history) != len(score_history):
        raise ValueError("archive_ce_history and archive_score_history must have same length")

    pred_history = None
    if "archive_pred_history" in payload:
        pred_history = ensure_1d_list(payload["archive_pred_history"], "archive_pred_history")
        if len(pred_history) != len(ce_history):
            pred_history = None

    out_dir.mkdir(parents=True, exist_ok=True)

    original_label = summary.get("original_label", None)
    if original_label is not None:
        original_label = int(original_label)

    score_objective = str(summary.get("score_objective", "min"))
    if score_objective not in {"min", "max"}:
        score_objective = "min"

    frame_paths = save_frames(
        ce_history=ce_history,
        score_history=score_history,
        pred_history=pred_history,
        output_dir=out_dir,
        original_label=original_label,
        score_objective=score_objective,
        dpi=args.dpi,
    )

    best_score_curve = save_best_score_curve(
        score_history=score_history,
        score_objective=score_objective,
        output_path=out_dir / "best_score_per_iteration.png",
        dpi=args.dpi,
    )
    np.save(out_dir / "best_score_per_iteration.npy", best_score_curve)

    best_ce_curve = save_best_ce_curve(
        ce_history=ce_history,
        output_path=out_dir / "best_ce_per_iteration.png",
        dpi=args.dpi,
    )
    np.save(out_dir / "best_ce_per_iteration.npy", best_ce_curve)
    save_iteration_csv(
        out_csv=out_dir / "best_metrics_per_iteration.csv",
        best_ce_curve=best_ce_curve,
        best_score_curve=best_score_curve,
    )

    gif_path = out_dir / "pareto_front_evolution.gif"
    gif_saved = False
    save_gif = not bool(args.no_gif)
    if save_gif:
        gif_saved = export_gif(frame_paths, gif_path, fps=args.fps)

    final_maps_report = None
    if args.export_final_maps:
        final_archive_path = args.final_archive_adversarials
        if final_archive_path is None:
            final_archive_path = final_archive_default_path
        if final_archive_path is None:
            raise ValueError(
                "final archive path is required for --export_final_maps "
                "(provide --final_archive_adversarials or keep <sample_dir>/final_archive_adversarials.npy)"
            )

        model_name = args.model_name or str(summary.get("model_name", "")).strip()
        if not model_name:
            raise ValueError("model_name is required for --export_final_maps (pass --model_name or provide summary.json)")

        if original_label is None:
            raise ValueError("original_label is required for --export_final_maps (provide summary.json with original_label)")

        final_maps_report = export_final_archive_maps(
            final_archive_adversarials_path=final_archive_path,
            output_dir=out_dir,
            model_name=model_name,
            gt_label=int(original_label),
            batch_size=args.batch_size,
        )

    report = {
        "sample_dir": str(args.sample_dir),
        "archive_history": str(archive_path),
        "summary": str(summary_path) if summary_path is not None else None,
        "iterations": len(ce_history),
        "frames_dir": str(out_dir / "frames"),
        "best_ce_curve_png": str(out_dir / "best_ce_per_iteration.png"),
        "best_ce_curve_npy": str(out_dir / "best_ce_per_iteration.npy"),
        "best_score_curve_png": str(out_dir / "best_score_per_iteration.png"),
        "best_score_curve_npy": str(out_dir / "best_score_per_iteration.npy"),
        "best_metrics_csv": str(out_dir / "best_metrics_per_iteration.csv"),
        "gif_path": str(gif_path) if save_gif else None,
        "gif_saved": bool(gif_saved),
        "score_objective": score_objective,
        "final_maps": final_maps_report,
    }

    with open(out_dir / "visualization_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if save_gif and not gif_saved:
        print("GIF export failed because imageio is not available. Install imageio to enable GIF output.")


if __name__ == "__main__":
    main()
