from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

try:
    from bcos.data.transforms import AddInverse
except ModuleNotFoundError:
    AddInverse = None


@dataclass
class PGDResult:
    original_label: int
    best_ce: float
    best_score: float
    class_preserved: bool
    history_best_ce: List[float]
    history_best_score: List[float]
    history_ce: List[float]
    history_score: List[float]
    history_descriptor_1: List[float]
    history_descriptor_2: List[float]
    history_iou: List[float]
    history_dice: List[float]
    history_pred: List[int]
    used_ce_fallback: bool
    best_perturbation: np.ndarray
    best_adversarial: np.ndarray


class PGDNSGAScoreAttack:
    def __init__(
        self,
        model,
        device,
        num_iterations: int,
        epsilon: float,
        step_size: float,
        random_start_std: float = 0.5,
        score_mode: str = "mean",
        score_objective: str = "min",
        weight_descriptor_1: float = 0.5,
        weight_descriptor_2: float = 0.5,
        overlap_percentile: float = 95.0,
        weight_overlap_iou: float = 0.0,
        weight_overlap_dice: float = 0.0,
    ):
        self.model = model
        self.device = device
        self.num_iterations = int(num_iterations)
        self.epsilon = float(epsilon)
        self.step_size = float(step_size)
        self.random_start_std = float(random_start_std)
        self.score_mode = str(score_mode)
        self.score_objective = str(score_objective)
        self.weight_descriptor_1 = float(weight_descriptor_1)
        self.weight_descriptor_2 = float(weight_descriptor_2)
        self.overlap_percentile = float(overlap_percentile)
        self.weight_overlap_iou = float(weight_overlap_iou)
        self.weight_overlap_dice = float(weight_overlap_dice)

        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if self.epsilon > 1.0:
            self.epsilon = self.epsilon / 255.0
        if self.step_size <= 0:
            raise ValueError("step_size must be > 0")
        if self.step_size > 1.0:
            self.step_size = self.step_size / 255.0
        if self.score_mode not in {"descriptor_1", "descriptor_2", "mean"}:
            raise ValueError("score_mode must be one of: 'descriptor_1', 'descriptor_2', 'mean'")
        if self.score_objective not in {"min", "max"}:
            raise ValueError("score_objective must be one of: 'min', 'max'")
        if not (0.0 < self.overlap_percentile < 100.0):
            raise ValueError("overlap_percentile must be in (0, 100)")

        self._split_transform()

    def _split_transform(self) -> None:
        bcos_transform_class = self.model.transform
        self.spatial_transform = transforms.Compose(
            bcos_transform_class.transforms.transforms[:-1]
        )
        self.bcos_transform = transforms.Compose(
            bcos_transform_class.transforms.transforms[-1:]
        )

        self._normalize_mean = None
        self._normalize_std = None
        self._use_add_inverse = False
        for t in self.bcos_transform.transforms:
            if isinstance(t, transforms.Normalize):
                self._normalize_mean = torch.as_tensor(
                    t.mean, dtype=torch.float32, device=self.device
                ).view(1, -1, 1, 1)
                self._normalize_std = torch.as_tensor(
                    t.std, dtype=torch.float32, device=self.device
                ).view(1, -1, 1, 1)
            elif AddInverse is not None and isinstance(t, AddInverse):
                self._use_add_inverse = True

    def _apply_tail_transform_from_scaled(self, batch_scaled_nchw: torch.Tensor) -> torch.Tensor:
        batch = batch_scaled_nchw.to(device=self.device, dtype=torch.float32)
        batch = torch.clamp(batch, 0.0, 1.0)

        if self._normalize_mean is not None and self._normalize_std is not None:
            return (batch - self._normalize_mean) / self._normalize_std
        if self._use_add_inverse:
            return torch.cat([batch, 1.0 - batch], dim=1)
        return batch

    def _to_model_batch(self, scaled_nchw: torch.Tensor) -> torch.Tensor:
        return self._apply_tail_transform_from_scaled(scaled_nchw)

    @staticmethod
    def _to_numpy(arr_like) -> np.ndarray:
        if isinstance(arr_like, torch.Tensor):
            return arr_like.detach().cpu().numpy().copy()
        return np.asarray(arr_like).copy()

    @staticmethod
    def _chw_float_to_hwc_uint8(x: np.ndarray) -> np.ndarray:
        y = np.transpose(np.asarray(x, dtype=np.float32), (1, 2, 0))
        y = np.clip(np.round(y * 255.0), 0, 255)
        return y.astype(np.uint8, copy=True)

    @staticmethod
    def _chw_float_to_hwc_float32(x: np.ndarray) -> np.ndarray:
        y = np.transpose(np.asarray(x, dtype=np.float32), (1, 2, 0))
        return y.astype(np.float32, copy=True)

    def _evaluate_single(
        self,
        adv_scaled_nchw: torch.Tensor,
        gt_explain_map: torch.Tensor,
        gt_contribution_map: torch.Tensor,
        gt_label: int,
    ) -> Dict[str, torch.Tensor]:
        model_batch = self._to_model_batch(adv_scaled_nchw)
        if not model_batch.requires_grad:
            model_batch = model_batch.requires_grad_(True)

        with torch.enable_grad(), self.model.explanation_mode():
            logits = self.model(model_batch)
            target_logit = logits[:, int(gt_label)]
            dynamic_linear_weights = torch.autograd.grad(
                target_logit.sum(),
                model_batch,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if dynamic_linear_weights is None:
                dynamic_linear_weights = torch.zeros_like(model_batch)

        contribution_maps = (model_batch * dynamic_linear_weights).sum(dim=1)
        explain_maps = self._gradient_to_image_batch(
            model_batch,
            dynamic_linear_weights,
            smooth=15,
            alpha_percentile=99.5,
        )

        if not isinstance(gt_explain_map, torch.Tensor):
            gt_explain_map = torch.as_tensor(gt_explain_map, device=self.device)
        else:
            gt_explain_map = gt_explain_map.to(self.device)

        if not isinstance(gt_contribution_map, torch.Tensor):
            gt_contribution_map = torch.as_tensor(gt_contribution_map, device=self.device)
        else:
            gt_contribution_map = gt_contribution_map.to(self.device)

        if gt_explain_map.dim() == 3:
            gt_explain_map = gt_explain_map.unsqueeze(0)
        if gt_contribution_map.dim() == 2:
            gt_contribution_map = gt_contribution_map.unsqueeze(0)

        explain_flat = explain_maps.reshape(explain_maps.shape[0], -1)
        gt_explain_flat = gt_explain_map.reshape(gt_explain_map.shape[0], -1)
        if gt_explain_flat.shape[0] == 1 and explain_flat.shape[0] > 1:
            gt_explain_flat = gt_explain_flat.expand(explain_flat.shape[0], -1)
        if explain_flat.shape != gt_explain_flat.shape:
            raise RuntimeError(
                "Explanation shape mismatch: "
                f"explain_maps={tuple(explain_maps.shape)}, gt_explain_map={tuple(gt_explain_map.shape)}, "
                f"explain_flat={tuple(explain_flat.shape)}, gt_explain_flat={tuple(gt_explain_flat.shape)}"
            )
        descriptor_1 = F.cosine_similarity(explain_flat, gt_explain_flat, dim=1, eps=1e-8)

        iou_bin, dice_bin, soft_iou, soft_dice = self._compute_overlap_metrics(
            explain_maps=explain_maps,
            gt_explain_map=gt_explain_map,
            percentile=self.overlap_percentile,
        )

        contribution_flat = contribution_maps.reshape(contribution_maps.shape[0], -1)
        gt_contribution_flat = gt_contribution_map.reshape(gt_contribution_map.shape[0], -1)
        if gt_contribution_flat.shape[0] == 1 and contribution_flat.shape[0] > 1:
            gt_contribution_flat = gt_contribution_flat.expand(contribution_flat.shape[0], -1)
        if contribution_flat.shape != gt_contribution_flat.shape:
            raise RuntimeError(
                "Contribution shape mismatch: "
                f"contribution_maps={tuple(contribution_maps.shape)}, gt_contribution_map={tuple(gt_contribution_map.shape)}, "
                f"contribution_flat={tuple(contribution_flat.shape)}, gt_contribution_flat={tuple(gt_contribution_flat.shape)}"
            )
        descriptor_2 = F.cosine_similarity(contribution_flat, gt_contribution_flat, dim=1, eps=1e-8)
        descriptor_2 = torch.clamp(descriptor_2, 0.0, 1.0)

        if self.score_mode == "descriptor_1":
            score = descriptor_1
        elif self.score_mode == "descriptor_2":
            score = descriptor_2
        else:
            score = 0.5 * (descriptor_1 + descriptor_2)

        targets = torch.full((logits.shape[0],), int(gt_label), dtype=torch.long, device=self.device)
        ce = F.cross_entropy(logits, targets, reduction="none")
        pred = logits.argmax(dim=1)

        weighted = self.weight_descriptor_1 * descriptor_1 + self.weight_descriptor_2 * descriptor_2
        if self.score_objective == "min":
            loss = weighted.mean()
        else:
            loss = (-weighted).mean()

        # Minimize overlap between binary-important regions of explanation maps.
        loss = loss + self.weight_overlap_iou * soft_iou.mean() + self.weight_overlap_dice * soft_dice.mean()

        return {
            "loss": loss,
            "ce": ce,
            "score": score,
            "descriptor_1": descriptor_1,
            "descriptor_2": descriptor_2,
            "iou": iou_bin,
            "dice": dice_bin,
            "pred": pred,
        }

    @staticmethod
    def _to_importance_map(explain_maps: torch.Tensor) -> torch.Tensor:
        # Input expected as NHWC. Use alpha when available, else RGB norm.
        if explain_maps.ndim != 4:
            raise ValueError(f"Expected explain maps as 4D tensor, got shape={tuple(explain_maps.shape)}")
        if explain_maps.shape[-1] >= 4:
            importance = explain_maps[..., 3]
        else:
            importance = torch.norm(explain_maps[..., : min(3, explain_maps.shape[-1])], p=2, dim=-1)
        return importance.to(dtype=torch.float32)

    def _compute_overlap_metrics(
        self,
        explain_maps: torch.Tensor,
        gt_explain_map: torch.Tensor,
        percentile: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = 1e-8
        cur_imp = self._to_importance_map(explain_maps)
        gt_imp = self._to_importance_map(gt_explain_map)

        if gt_imp.shape[0] == 1 and cur_imp.shape[0] > 1:
            gt_imp = gt_imp.expand(cur_imp.shape[0], -1, -1)

        if cur_imp.shape != gt_imp.shape:
            raise RuntimeError(
                "Importance map mismatch: "
                f"cur_imp={tuple(cur_imp.shape)}, gt_imp={tuple(gt_imp.shape)}"
            )

        q = float(percentile) / 100.0
        cur_flat = cur_imp.flatten(start_dim=1)
        gt_flat = gt_imp.flatten(start_dim=1)

        cur_thr = torch.quantile(cur_flat, q=q, dim=1, keepdim=True)
        gt_thr = torch.quantile(gt_flat, q=q, dim=1, keepdim=True)

        cur_mask = (cur_flat >= cur_thr).to(dtype=torch.float32)
        gt_mask = (gt_flat >= gt_thr).to(dtype=torch.float32)

        inter = (cur_mask * gt_mask).sum(dim=1)
        union = (cur_mask + gt_mask - cur_mask * gt_mask).sum(dim=1)
        iou_bin = inter / (union + eps)

        denom = cur_mask.sum(dim=1) + gt_mask.sum(dim=1)
        dice_bin = (2.0 * inter) / (denom + eps)

        # Soft surrogate for optimization stability.
        k = 20.0
        cur_thr_soft = cur_thr.detach()
        gt_thr_soft = gt_thr.detach()
        cur_soft = torch.sigmoid(k * (cur_flat - cur_thr_soft))
        gt_soft = torch.sigmoid(k * (gt_flat - gt_thr_soft))

        inter_soft = (cur_soft * gt_soft).sum(dim=1)
        union_soft = (cur_soft + gt_soft - cur_soft * gt_soft).sum(dim=1)
        iou_soft = inter_soft / (union_soft + eps)

        denom_soft = cur_soft.sum(dim=1) + gt_soft.sum(dim=1)
        dice_soft = (2.0 * inter_soft) / (denom_soft + eps)

        return iou_bin, dice_bin, iou_soft, dice_soft

    def _gradient_to_image_batch(
        self,
        image: torch.Tensor,
        linear_mapping: torch.Tensor,
        smooth: int = 15,
        alpha_percentile: float = 99.5,
    ) -> torch.Tensor:
        # image/linear_mapping: [N, C, H, W], typically C=6 for B-cos.
        contribs = (image * linear_mapping).sum(dim=1, keepdim=True)

        channel_max = linear_mapping.abs().amax(dim=1, keepdim=True) + 1e-12
        rgb_grad = linear_mapping / channel_max
        rgb_grad = rgb_grad.clamp(min=0)

        if rgb_grad.shape[1] >= 6:
            rgb_num = rgb_grad[:, :3]
            rgb_den = rgb_grad[:, :3] + rgb_grad[:, 3:6] + 1e-12
            rgb_grad = rgb_num / rgb_den
        else:
            rgb_grad = rgb_grad[:, : min(3, rgb_grad.shape[1])]
            if rgb_grad.shape[1] == 1:
                rgb_grad = rgb_grad.repeat(1, 3, 1, 1)
            elif rgb_grad.shape[1] == 2:
                rgb_grad = torch.cat([rgb_grad, rgb_grad[:, :1]], dim=1)

        alpha = linear_mapping.norm(p=2, dim=1, keepdim=True)
        alpha = torch.where(contribs < 0, torch.full_like(alpha, 1e-12), alpha)
        if smooth > 0:
            alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)

        q = torch.quantile(
            alpha.flatten(start_dim=1),
            q=float(alpha_percentile) / 100.0,
            dim=1,
            keepdim=True,
        ).view(-1, 1, 1, 1)
        alpha = (alpha / (q + 1e-12)).clamp(0.0, 1.0)

        rgba = torch.cat([rgb_grad, alpha], dim=1)
        return rgba.permute(0, 2, 3, 1).contiguous()

    def run(self, img: Image.Image) -> PGDResult:
        img = self.spatial_transform(img)
        if not isinstance(img, torch.Tensor):
            raise TypeError("Expected spatial_transform to return tensor in [0,1]")
        if img.ndim != 3:
            raise ValueError(f"Expected CHW image tensor, got shape={tuple(img.shape)}")

        base_img = torch.clamp(img.to(self.device, dtype=torch.float32), 0.0, 1.0)

        clean_in = self._to_model_batch(base_img.unsqueeze(0)).detach().requires_grad_(True)
        original_results = self.model.explain(clean_in)
        original_explain_map = original_results["explanation"]
        original_contribution_map = original_results["contribution_map"]

        prediction = original_results["prediction"]
        if isinstance(prediction, torch.Tensor):
            original_label = int(prediction.reshape(-1)[0].item())
        elif isinstance(prediction, (list, tuple, np.ndarray)):
            original_label = int(prediction[0])
        else:
            original_label = int(prediction)

        delta = torch.randn_like(base_img) * (self.random_start_std * self.epsilon)
        delta = torch.clamp(delta, -self.epsilon, self.epsilon)

        history_best_ce: List[float] = []
        history_best_score: List[float] = []
        history_ce: List[float] = []
        history_score: List[float] = []
        history_descriptor_1: List[float] = []
        history_descriptor_2: List[float] = []
        history_iou: List[float] = []
        history_dice: List[float] = []
        history_pred: List[int] = []

        best_ce = float("inf")
        best_score = float("inf") if self.score_objective == "min" else float("-inf")
        best_adv = torch.clamp(base_img + delta, 0.0, 1.0).detach()
        best_delta = delta.detach().clone()

        class_preserved = False
        used_ce_fallback = False

        for _ in range(self.num_iterations):
            delta = delta.detach().clone().requires_grad_(True)
            adv = torch.clamp(base_img + delta, 0.0, 1.0)

            metrics = self._evaluate_single(
                adv.unsqueeze(0),
                original_explain_map,
                original_contribution_map,
                original_label,
            )

            loss = metrics["loss"]
            grad = torch.autograd.grad(
                loss,
                delta,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                used_ce_fallback = True
                ce_loss = metrics["ce"].mean()
                grad = torch.autograd.grad(
                    ce_loss,
                    delta,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )[0]
                if grad is None:
                    grad = torch.zeros_like(delta)

            with torch.no_grad():
                delta = delta - self.step_size * torch.sign(grad)
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adv = torch.clamp(base_img + delta, 0.0, 1.0)

                ce_val = float(metrics["ce"][0].item())
                score_val = float(metrics["score"][0].item())
                d1_val = float(metrics["descriptor_1"][0].item())
                d2_val = float(metrics["descriptor_2"][0].item())
                iou_val = float(metrics["iou"][0].item())
                dice_val = float(metrics["dice"][0].item())
                pred_val = int(metrics["pred"][0].item())

                class_preserved = class_preserved or (pred_val == int(original_label))

                if self.score_objective == "min":
                    better_score = score_val < best_score
                else:
                    better_score = score_val > best_score

                if better_score or (np.isclose(score_val, best_score) and ce_val < best_ce):
                    best_score = score_val
                    best_ce = ce_val
                    best_adv = adv.detach().clone()
                    best_delta = delta.detach().clone()

                history_ce.append(ce_val)
                history_score.append(score_val)
                history_descriptor_1.append(d1_val)
                history_descriptor_2.append(d2_val)
                history_iou.append(iou_val)
                history_dice.append(dice_val)
                history_pred.append(pred_val)

                if len(history_best_ce) == 0:
                    running_best_ce = ce_val
                    running_best_score = score_val
                else:
                    running_best_ce = min(history_best_ce[-1], ce_val)
                    if self.score_objective == "min":
                        running_best_score = min(history_best_score[-1], score_val)
                    else:
                        running_best_score = max(history_best_score[-1], score_val)

                history_best_ce.append(float(running_best_ce))
                history_best_score.append(float(running_best_score))

        best_adv_np = self._to_numpy(best_adv)
        best_delta_np = self._to_numpy(best_delta)

        return PGDResult(
            original_label=original_label,
            best_ce=float(best_ce),
            best_score=float(best_score),
            class_preserved=bool(class_preserved),
            history_best_ce=[float(v) for v in history_best_ce],
            history_best_score=[float(v) for v in history_best_score],
            history_ce=[float(v) for v in history_ce],
            history_score=[float(v) for v in history_score],
            history_descriptor_1=[float(v) for v in history_descriptor_1],
            history_descriptor_2=[float(v) for v in history_descriptor_2],
            history_iou=[float(v) for v in history_iou],
            history_dice=[float(v) for v in history_dice],
            history_pred=[int(v) for v in history_pred],
            used_ce_fallback=bool(used_ce_fallback),
            best_perturbation=self._chw_float_to_hwc_float32(best_delta_np),
            best_adversarial=self._chw_float_to_hwc_uint8(best_adv_np),
        )
