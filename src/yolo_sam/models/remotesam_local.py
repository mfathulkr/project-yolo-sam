from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from yolo_sam.models.sam3_local import resolve_torch_device


@dataclass
class RemoteSAMResult:
    merged_mask: np.ndarray
    score_map: np.ndarray
    prompts: list[str]
    prompt_area_ratios: dict[str, float]
    prompt_score_means: dict[str, float]


class LocalRemoteSAMSegmenter:
    def __init__(
        self,
        external_root: Path,
        checkpoint_path: Path,
        device: str | int,
        use_epoc: bool = False,
    ) -> None:
        self.external_root = external_root.resolve()
        if str(self.external_root) not in sys.path:
            sys.path.insert(0, str(self.external_root))

        from tasks.code.model import RemoteSAM, init_demo_model  # type: ignore

        self.device = resolve_torch_device(device)
        old_argv = sys.argv[:]
        try:
            sys.argv = [old_argv[0]]
            base_model = init_demo_model(str(checkpoint_path), self.device)
        finally:
            sys.argv = old_argv
        self.model = RemoteSAM(base_model, self.device, use_EPOC=use_epoc)

    def segment(
        self,
        image: Image.Image,
        prompts: list[str],
        prob_threshold: float,
    ) -> RemoteSAMResult:
        masks, probs = self.model.semantic_seg(
            image=image,
            classnames=prompts,
            return_prob=True,
        )

        score_maps: list[np.ndarray] = []
        prompt_area_ratios: dict[str, float] = {}
        prompt_score_means: dict[str, float] = {}
        for prompt in prompts:
            score = np.asarray(probs[prompt], dtype=np.float32)
            score_maps.append(score)
            prompt_area_ratios[prompt] = float(np.asarray(masks[prompt]).astype(bool).mean())
            prompt_score_means[prompt] = float(score.mean())

        if score_maps:
            score_map = np.maximum.reduce(score_maps)
        else:
            score_map = np.zeros(image.size[::-1], dtype=np.float32)

        merged_mask = score_map >= float(prob_threshold)
        return RemoteSAMResult(
            merged_mask=merged_mask,
            score_map=score_map,
            prompts=prompts,
            prompt_area_ratios=prompt_area_ratios,
            prompt_score_means=prompt_score_means,
        )
