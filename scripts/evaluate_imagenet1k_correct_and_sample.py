import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

try:
    from attack.utils import load_model, seed_everything
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from attack.utils import load_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or multiple B-cos models on ImageNet1K val folder structure and "
            "save all correctly predicted samples, plus a 1000-class sampling file."
        )
    )
    parser.add_argument(
        "--imagenet_val_dir",
        type=Path,
        default=Path("/datastore/elo/quanphm/dataset/ImageNet1K/val"),
        help="Path to ImageNet1K validation directory (contains class-key subfolders).",
    )
    parser.add_argument(
        "--class_index_json",
        type=Path,
        default=Path("data/imagenet_1k_label"),
        help=(
            "Path to class-index mapping file used to map class folder key to label index. "
            "Supports formats like {\"n014...\": [0, \"tench\"]} or {\"0\": [\"n014...\", \"tench\"]}."
        ),
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        required=True,
        help="Model names accepted by attack.utils.load_model (example: resnet18 resnet50).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/imagenet_eval"),
        help="Directory where output json files are saved.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Target number of sampled images (typically 1000, one image per class).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for debugging (evaluate first N samples only).",
    )
    return parser.parse_args()


def load_class_key_to_idx(json_path: Path) -> Dict[str, int]:
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping: Dict[str, int] = {}

    if isinstance(raw, list):
        for i, item in enumerate(raw):
            if isinstance(item, str):
                mapping[item] = i
            elif isinstance(item, dict):
                key = str(item.get("key", item.get("wnid", i)))
                mapping[key] = int(item.get("index", i))
        return mapping

    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported class-index json format: {type(raw)}")

    for k, v in raw.items():
        if str(k).isdigit():
            idx = int(k)
            mapping[str(k)] = idx
            if isinstance(v, (list, tuple)) and len(v) > 0:
                mapping[str(v[0])] = idx
            elif isinstance(v, dict):
                if "wnid" in v:
                    mapping[str(v["wnid"])] = idx
                if "key" in v:
                    mapping[str(v["key"])] = idx
        else:
            if isinstance(v, int):
                mapping[str(k)] = int(v)
            elif isinstance(v, str) and v.isdigit():
                mapping[str(k)] = int(v)
            elif isinstance(v, (list, tuple)) and len(v) > 0:
                # Common format in this repo: {"n01440764": [0, "tench"]}
                mapping[str(k)] = int(v[0])
            elif isinstance(v, dict) and "index" in v:
                mapping[str(k)] = int(v["index"])

    if not mapping:
        raise ValueError("Could not build class-key mapping from the provided json file.")
    return mapping


def resolve_label_index(class_key: str, key_to_idx: Dict[str, int]) -> Optional[int]:
    if class_key in key_to_idx:
        return int(key_to_idx[class_key])
    if class_key.isdigit():
        return int(class_key)
    return None


def collect_samples(
    imagenet_val_dir: Path,
    key_to_idx: Dict[str, int],
    max_samples: Optional[int],
) -> List[Tuple[str, int, Path]]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples: List[Tuple[str, int, Path]] = []

    class_dirs = sorted([p for p in imagenet_val_dir.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        class_key = class_dir.name
        gt_idx = resolve_label_index(class_key, key_to_idx)
        if gt_idx is None:
            continue

        for img_path in sorted(class_dir.iterdir()):
            if not img_path.is_file() or img_path.suffix.lower() not in valid_ext:
                continue
            samples.append((class_key, gt_idx, img_path))
            if max_samples is not None and len(samples) >= max_samples:
                return samples

    return samples


def _forward_logits(model: torch.nn.Module, input_batch: torch.Tensor) -> torch.Tensor:
    output = model(input_batch)
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported model output type: {type(output)}")


def evaluate_model(
    model_name: str,
    model: torch.nn.Module,
    samples: List[Tuple[str, int, Path]],
    batch_size: int,
) -> Dict[str, object]:
    transform = model.transform
    device = next(model.parameters()).device

    total = len(samples)
    num_correct = 0
    correct_records: List[Dict[str, object]] = []

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, total, batch_size), desc=f"Evaluating {model_name}"):
            batch_meta = samples[start : start + batch_size]
            batch_tensors = []
            for class_key, gt_idx, img_path in batch_meta:
                img = Image.open(img_path).convert("RGB")
                batch_tensors.append(transform(img))

            input_batch = torch.stack(batch_tensors, dim=0).to(device)
            logits = _forward_logits(model, input_batch)
            preds = logits.argmax(dim=1).detach().cpu().tolist()

            for (class_key, gt_idx, img_path), pred in zip(batch_meta, preds):
                if int(pred) == int(gt_idx):
                    num_correct += 1
                    correct_records.append(
                        {
                            "image_path": str(img_path),
                            "pred": int(pred),
                            "key": class_key,
                        }
                    )

    accuracy = float(num_correct / total) if total > 0 else 0.0
    return {
        "num_samples": total,
        "num_correct": num_correct,
        "accuracy": accuracy,
        "correct_predictions": correct_records,
    }


def sample_one_per_class(
    records: List[Dict[str, object]],
    sample_size: int,
    seed: int,
) -> Dict[str, object]:
    by_class: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        by_class[str(record["key"])].append(record)

    rng = random.Random(seed)
    all_keys = sorted(by_class.keys())

    selected_keys = all_keys
    if len(all_keys) > sample_size:
        selected_keys = sorted(rng.sample(all_keys, sample_size))

    sampled_records: List[Dict[str, object]] = []
    for key in selected_keys:
        sampled_records.append(rng.choice(by_class[key]))

    return {
        "num_classes_with_correct": len(all_keys),
        "num_sampled": len(sampled_records),
        "sampled_records": sampled_records,
    }


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    if not args.imagenet_val_dir.exists():
        raise FileNotFoundError(f"imagenet_val_dir not found: {args.imagenet_val_dir}")
    if not args.class_index_json.exists():
        raise FileNotFoundError(f"class_index_json not found: {args.class_index_json}")

    key_to_idx = load_class_key_to_idx(args.class_index_json)
    samples = collect_samples(args.imagenet_val_dir, key_to_idx, args.max_samples)
    if len(samples) == 0:
        raise RuntimeError("No valid samples found. Check val dir structure and class-index mapping.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    evaluate_payload: Dict[str, object] = {
        "imagenet_val_dir": str(args.imagenet_val_dir),
        "class_index_json": str(args.class_index_json),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "num_total_samples": int(len(samples)),
        "models": {},
    }

    sampled_payload: Dict[str, object] = {
        "source": "evaluate_imagenet1k_correct_and_sample.py",
        "target_sample_size": int(args.sample_size),
        "models": {},
    }

    for model_name in args.model_names:
        model = load_model(model_name)
        metrics = evaluate_model(model_name, model, samples, args.batch_size)
        evaluate_payload["models"][model_name] = metrics

        sampled = sample_one_per_class(
            records=metrics["correct_predictions"],
            sample_size=args.sample_size,
            seed=args.seed,
        )
        sampled_payload["models"][model_name] = sampled

    eval_json_path = args.output_dir / "correct_predictions_by_model.json"
    sample_json_path = args.output_dir / "sampled_1000_from_1000_classes_by_model.json"

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(evaluate_payload, f, indent=2, ensure_ascii=False)

    with open(sample_json_path, "w", encoding="utf-8") as f:
        json.dump(sampled_payload, f, indent=2, ensure_ascii=False)

    print(f"Saved: {eval_json_path}")
    print(f"Saved: {sample_json_path}")


if __name__ == "__main__":
    main(parse_args())
