import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunRecord:
    model_name: str
    run_dir: Path
    history_best_ce: np.ndarray
    history_best_score: np.ndarray
    class_preserved: bool
    best_ce: float
    best_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate NSGA results over many samples (e.g., 1000 images), then export "
            "mean±std curves, CSV files, and ASR/summary statistics."
        )
    )
    parser.add_argument(
        "--runs_root",
        type=Path,
        required=True,
        help="Root folder produced by run_main_nsga_attack_sample.py",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output folder (default: <runs_root>/aggregate_visualization)",
    )
    parser.add_argument(
        "--score_objective",
        type=str,
        default="min",
        choices=["min", "max"],
        help="How to interpret score improvement and threshold checks.",
    )
    parser.add_argument(
        "--score_success_threshold",
        type=float,
        default=None,
        help="Optional ASR threshold on final best_score.",
    )
    parser.add_argument(
        "--ce_success_threshold",
        type=float,
        default=None,
        help="Optional ASR threshold on final best_ce (<= threshold means success).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def _load_history_array(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"Empty history array at: {path}")
    return arr


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid summary format at {path}")
    return data


def discover_runs(runs_root: Path) -> List[RunRecord]:
    summary_paths = sorted(runs_root.rglob("summary.json"))
    records: List[RunRecord] = []

    for summary_path in summary_paths:
        run_dir = summary_path.parent
        ce_path = run_dir / "history_best_ce.npy"
        score_path = run_dir / "history_best_score.npy"
        if not ce_path.exists() or not score_path.exists():
            continue

        summary = _load_summary(summary_path)
        model_name = str(summary.get("model_name", "unknown_model"))

        try:
            history_best_ce = _load_history_array(ce_path)
            history_best_score = _load_history_array(score_path)
        except Exception:
            continue

        if history_best_ce.shape[0] != history_best_score.shape[0]:
            min_len = min(history_best_ce.shape[0], history_best_score.shape[0])
            history_best_ce = history_best_ce[:min_len]
            history_best_score = history_best_score[:min_len]

        if history_best_ce.size == 0:
            continue

        records.append(
            RunRecord(
                model_name=model_name,
                run_dir=run_dir,
                history_best_ce=history_best_ce,
                history_best_score=history_best_score,
                class_preserved=bool(summary.get("class_preserved", False)),
                best_ce=float(summary.get("best_ce", float(history_best_ce[-1]))),
                best_score=float(summary.get("best_score", float(history_best_score[-1]))),
            )
        )

    return records


def stack_with_nan_padding(histories: List[np.ndarray]) -> np.ndarray:
    max_len = max(arr.shape[0] for arr in histories)
    stacked = np.full((len(histories), max_len), np.nan, dtype=np.float32)
    for i, arr in enumerate(histories):
        stacked[i, : arr.shape[0]] = arr
    return stacked


def nanmean_std(stacked: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    count = np.sum(np.isfinite(stacked), axis=0).astype(np.int64)
    return mean.astype(np.float32), std.astype(np.float32), count


def save_curve_plot(
    out_path: Path,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    count: np.ndarray,
    title: str,
    y_label: str,
    color: str,
    dpi: int,
) -> None:
    x = np.arange(y_mean.shape[0], dtype=np.int64)
    low = y_mean - y_std
    high = y_mean + y_std

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=dpi)
    ax.plot(x, y_mean, color=color, linewidth=2, label="mean")
    ax.fill_between(x, low, high, color=color, alpha=0.2, label="mean ± std")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    ax2 = ax.twinx()
    ax2.plot(x, count, color="#666666", linewidth=1.2, alpha=0.7, label="valid runs")
    ax2.set_ylabel("Valid runs")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_curve_csv(
    out_path: Path,
    ce_mean: np.ndarray,
    ce_std: np.ndarray,
    ce_count: np.ndarray,
    score_mean: np.ndarray,
    score_std: np.ndarray,
    score_count: np.ndarray,
) -> None:
    n = max(ce_mean.shape[0], score_mean.shape[0])

    def _safe(arr: np.ndarray, idx: int, cast=float):
        if idx >= arr.shape[0]:
            return ""
        return cast(arr[idx])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "iteration",
                "best_ce_mean",
                "best_ce_std",
                "best_ce_valid_runs",
                "best_score_mean",
                "best_score_std",
                "best_score_valid_runs",
            ]
        )
        for i in range(n):
            writer.writerow(
                [
                    i,
                    _safe(ce_mean, i, float),
                    _safe(ce_std, i, float),
                    _safe(ce_count, i, int),
                    _safe(score_mean, i, float),
                    _safe(score_std, i, float),
                    _safe(score_count, i, int),
                ]
            )


def compute_asr_metrics(
    records: List[RunRecord],
    score_objective: str,
    score_success_threshold: Optional[float],
    ce_success_threshold: Optional[float],
) -> Dict[str, object]:
    n = len(records)
    if n == 0:
        return {
            "num_runs": 0,
            "asr_class_preserved": 0.0,
            "asr_score_improved": 0.0,
            "asr_joint_preserved_score_improved": 0.0,
            "asr_score_threshold": None,
            "asr_joint_preserved_score_threshold": None,
            "asr_ce_threshold": None,
            "asr_joint_preserved_ce_threshold": None,
        }

    class_preserved = np.asarray([r.class_preserved for r in records], dtype=bool)

    init_score = np.asarray([float(r.history_best_score[0]) for r in records], dtype=np.float32)
    final_score = np.asarray([float(r.history_best_score[-1]) for r in records], dtype=np.float32)
    if score_objective == "min":
        score_improved = final_score < init_score
    else:
        score_improved = final_score > init_score

    final_ce = np.asarray([float(r.history_best_ce[-1]) for r in records], dtype=np.float32)

    result: Dict[str, object] = {
        "num_runs": n,
        "asr_class_preserved": float(np.mean(class_preserved)),
        "asr_score_improved": float(np.mean(score_improved)),
        "asr_joint_preserved_score_improved": float(np.mean(class_preserved & score_improved)),
    }

    if score_success_threshold is not None:
        if score_objective == "min":
            score_success = final_score <= float(score_success_threshold)
        else:
            score_success = final_score >= float(score_success_threshold)
        result["asr_score_threshold"] = float(np.mean(score_success))
        result["asr_joint_preserved_score_threshold"] = float(np.mean(class_preserved & score_success))
    else:
        result["asr_score_threshold"] = None
        result["asr_joint_preserved_score_threshold"] = None

    if ce_success_threshold is not None:
        ce_success = final_ce <= float(ce_success_threshold)
        result["asr_ce_threshold"] = float(np.mean(ce_success))
        result["asr_joint_preserved_ce_threshold"] = float(np.mean(class_preserved & ce_success))
    else:
        result["asr_ce_threshold"] = None
        result["asr_joint_preserved_ce_threshold"] = None

    return result


def compute_scalar_stats(records: List[RunRecord], score_objective: str) -> Dict[str, object]:
    if not records:
        return {}

    init_ce = np.asarray([float(r.history_best_ce[0]) for r in records], dtype=np.float32)
    final_ce = np.asarray([float(r.history_best_ce[-1]) for r in records], dtype=np.float32)
    init_score = np.asarray([float(r.history_best_score[0]) for r in records], dtype=np.float32)
    final_score = np.asarray([float(r.history_best_score[-1]) for r in records], dtype=np.float32)

    score_delta = final_score - init_score
    ce_delta = final_ce - init_ce

    if score_objective == "min":
        score_gain = init_score - final_score
    else:
        score_gain = final_score - init_score

    return {
        "best_ce": {
            "initial_mean": float(np.mean(init_ce)),
            "initial_std": float(np.std(init_ce)),
            "final_mean": float(np.mean(final_ce)),
            "final_std": float(np.std(final_ce)),
            "delta_mean": float(np.mean(ce_delta)),
            "delta_std": float(np.std(ce_delta)),
            "final_min": float(np.min(final_ce)),
            "final_max": float(np.max(final_ce)),
            "final_median": float(np.median(final_ce)),
        },
        "best_score": {
            "initial_mean": float(np.mean(init_score)),
            "initial_std": float(np.std(init_score)),
            "final_mean": float(np.mean(final_score)),
            "final_std": float(np.std(final_score)),
            "delta_mean": float(np.mean(score_delta)),
            "delta_std": float(np.std(score_delta)),
            "improvement_mean": float(np.mean(score_gain)),
            "improvement_std": float(np.std(score_gain)),
            "final_min": float(np.min(final_score)),
            "final_max": float(np.max(final_score)),
            "final_median": float(np.median(final_score)),
        },
    }


def write_group_outputs(
    records: List[RunRecord],
    group_name: str,
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    group_dir = output_root / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    ce_stack = stack_with_nan_padding([r.history_best_ce for r in records])
    score_stack = stack_with_nan_padding([r.history_best_score for r in records])

    ce_mean, ce_std, ce_count = nanmean_std(ce_stack)
    score_mean, score_std, score_count = nanmean_std(score_stack)

    save_curve_plot(
        out_path=group_dir / "curve_best_ce_mean_std.png",
        y_mean=ce_mean,
        y_std=ce_std,
        count=ce_count,
        title=f"[{group_name}] Best CE per Iteration (Mean ± Std)",
        y_label="Best CE (min)",
        color="#d62728",
        dpi=args.dpi,
    )

    score_label = "Best Score (min)" if args.score_objective == "min" else "Best Score (max)"
    save_curve_plot(
        out_path=group_dir / "curve_best_score_mean_std.png",
        y_mean=score_mean,
        y_std=score_std,
        count=score_count,
        title=f"[{group_name}] Best Score per Iteration (Mean ± Std)",
        y_label=score_label,
        color="#1f77b4",
        dpi=args.dpi,
    )

    save_curve_csv(
        out_path=group_dir / "curve_iteration_stats.csv",
        ce_mean=ce_mean,
        ce_std=ce_std,
        ce_count=ce_count,
        score_mean=score_mean,
        score_std=score_std,
        score_count=score_count,
    )

    asr_metrics = compute_asr_metrics(
        records=records,
        score_objective=args.score_objective,
        score_success_threshold=args.score_success_threshold,
        ce_success_threshold=args.ce_success_threshold,
    )
    scalar_stats = compute_scalar_stats(records=records, score_objective=args.score_objective)

    payload = {
        "group_name": group_name,
        "num_runs": len(records),
        "score_objective": args.score_objective,
        "score_success_threshold": args.score_success_threshold,
        "ce_success_threshold": args.ce_success_threshold,
        "asr": asr_metrics,
        "scalar_stats": scalar_stats,
    }

    with open(group_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(group_dir / "run_dirs.txt", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(str(rec.run_dir) + "\n")

    return payload


def main(args: argparse.Namespace) -> None:
    if not args.runs_root.exists() or not args.runs_root.is_dir():
        raise FileNotFoundError(f"runs_root is not a valid directory: {args.runs_root}")

    output_root = args.output_dir or (args.runs_root / "aggregate_visualization")
    output_root.mkdir(parents=True, exist_ok=True)

    records = discover_runs(args.runs_root)
    if not records:
        raise RuntimeError("No valid NSGA run folders found (missing summary/history files).")

    records_by_model: Dict[str, List[RunRecord]] = {}
    for rec in records:
        records_by_model.setdefault(rec.model_name, []).append(rec)

    index_payload = {
        "runs_root": str(args.runs_root),
        "output_root": str(output_root),
        "num_total_runs": len(records),
        "models": {},
        "overall": None,
    }

    overall_summary = write_group_outputs(
        records=records,
        group_name="overall",
        output_root=output_root,
        args=args,
    )
    index_payload["overall"] = overall_summary

    for model_name, model_records in sorted(records_by_model.items()):
        summary = write_group_outputs(
            records=model_records,
            group_name=f"model_{model_name}",
            output_root=output_root,
            args=args,
        )
        index_payload["models"][model_name] = summary

    with open(output_root / "index_summary.json", "w", encoding="utf-8") as f:
        json.dump(index_payload, f, indent=2, ensure_ascii=False)

    print(f"Discovered runs: {len(records)}")
    print(f"Saved aggregate outputs to: {output_root}")
    print("Generated groups: overall + per-model")


if __name__ == "__main__":
    main(parse_args())
