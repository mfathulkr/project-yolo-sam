from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam3_bbox_study.models.sam3_local import resolve_torch_device


@dataclass
class SegEarthOV3Result:
    merged_mask: np.ndarray
    score_map: np.ndarray
    prompts: list[str]
    presence_scores: list[float]
    instance_counts: list[int]
    instance_score_means: list[float]


class LocalSegEarthOV3Segmenter:
    """Minimal SegEarth-OV3-style SAM3 semantic/instance fusion wrapper.

    The upstream SegEarth-OV3 project is an mmseg segmentor. For this study we
    reuse its SAM3 processor and fusion logic directly so the baseline can run
    as a single-image open-vocabulary segmenter on CPU.
    """

    def __init__(
        self,
        external_root: Path,
        checkpoint_path: Path,
        bpe_path: Path,
        device: str | int,
        confidence_threshold: float,
        resolution: int = 1008,
    ) -> None:
        self.external_root = external_root.resolve()
        if str(self.external_root) not in sys.path:
            sys.path.insert(0, str(self.external_root))

        from sam3 import build_sam3_image_model  # type: ignore
        from sam3.model.sam3_image_processor import Sam3Processor  # type: ignore

        self.device = resolve_torch_device(device)
        self.model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            checkpoint_path=str(checkpoint_path),
            device=self.device,
            eval_mode=True,
            load_from_HF=False,
        )
        self.model.eval()
        self.processor = Sam3Processor(
            self.model,
            resolution=int(resolution),
            device=self.device,
            confidence_threshold=float(confidence_threshold),
        )

    def segment(
        self,
        image: Image.Image,
        prompts: list[str],
        prob_threshold: float,
        use_semantic_head: bool = True,
        use_instance_head: bool = True,
        use_presence_score: bool = True,
    ) -> SegEarthOV3Result:
        width, height = image.size
        merged_scores = torch.zeros((height, width), device=self.device, dtype=torch.float32)
        presence_scores: list[float] = []
        instance_counts: list[int] = []
        instance_score_means: list[float] = []

        with torch.inference_mode():
            state = self.processor.set_image(image)
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                state = self.processor.set_text_prompt(prompt=prompt, state=state)
                prompt_scores = torch.zeros((height, width), device=self.device, dtype=torch.float32)

                if use_instance_head:
                    masks = state.get("masks_logits")
                    scores = state.get("object_score")
                    if masks is not None and masks.shape[0] > 0:
                        instance_counts.append(int(masks.shape[0]))
                        if scores is not None and scores.numel() > 0:
                            instance_score_means.append(float(scores.detach().mean().cpu()))
                        else:
                            instance_score_means.append(0.0)

                        for instance_index in range(int(masks.shape[0])):
                            instance_score = 1.0
                            if scores is not None and scores.numel() > instance_index:
                                instance_score = float(scores[instance_index].detach().cpu())
                            instance_mask = masks[instance_index].squeeze().float()
                            if instance_mask.shape != (height, width):
                                instance_mask = F.interpolate(
                                    instance_mask.view(1, 1, *instance_mask.shape),
                                    size=(height, width),
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze()
                            prompt_scores = torch.maximum(prompt_scores, instance_mask * instance_score)
                    else:
                        instance_counts.append(0)
                        instance_score_means.append(0.0)

                if use_semantic_head:
                    semantic_scores = state.get("semantic_mask_logits")
                    if semantic_scores is not None:
                        semantic_scores = semantic_scores.squeeze().float()
                        if semantic_scores.shape != (height, width):
                            semantic_scores = F.interpolate(
                                semantic_scores.view(1, 1, *semantic_scores.shape),
                                size=(height, width),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()
                        prompt_scores = torch.maximum(prompt_scores, semantic_scores)

                presence = state.get("presence_score")
                presence_value = float(presence.detach().cpu()) if torch.is_tensor(presence) else float(presence or 1.0)
                presence_scores.append(presence_value)
                if use_presence_score:
                    prompt_scores = prompt_scores * presence_value

                merged_scores = torch.maximum(merged_scores, prompt_scores)

        score_map = merged_scores.detach().cpu().numpy()
        merged_mask = score_map >= float(prob_threshold)
        return SegEarthOV3Result(
            merged_mask=merged_mask,
            score_map=score_map,
            prompts=prompts,
            presence_scores=presence_scores,
            instance_counts=instance_counts,
            instance_score_means=instance_score_means,
        )
