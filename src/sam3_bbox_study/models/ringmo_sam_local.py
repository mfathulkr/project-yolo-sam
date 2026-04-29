from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import yaml


@dataclass
class RingMoSamResult:
    class_map: np.ndarray
    merged_mask: np.ndarray
    input_boxes: list[list[float]]
    class_ids: list[int]


class LocalRingMoSamSegmenter:
    def __init__(
        self,
        model_root: Path,
        config_path: Path,
        checkpoint_path: Path,
        device: str = "cpu",
        normalize: bool = True,
        class_ids: list[int] | None = None,
    ) -> None:
        self.model_root = model_root
        self.device = torch.device(device)
        self.normalize = normalize
        self.class_ids = class_ids or [5]

        model_root_resolved = str(model_root.resolve())
        if model_root_resolved not in sys.path:
            sys.path.insert(0, model_root_resolved)

        import models  # type: ignore[import-not-found]

        config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
        self.model = models.make(config["model"]).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

        steps: list[object] = [transforms.Resize([1024, 1024]), transforms.ToTensor()]
        if normalize:
            steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(steps)

    def predict_class_map(self, image: Image.Image) -> np.ndarray:
        input_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            image_embedding = self.model.image_encoder(input_tensor)
            sparse_embeddings, dense_embeddings, _ = self.model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                scatter=None,
            )
            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe().to(self.device),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            logits = self.model.postprocess_masks(low_res_masks, self.model.inp_size, self.model.inp_size)
            class_map = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return class_map

    @staticmethod
    def boxes_to_mask(boxes: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        mask = np.zeros((height, width), dtype=bool)
        for box in boxes:
            x1, y1, x2, y2 = box
            x1_i = max(0, min(width, int(np.floor(x1))))
            y1_i = max(0, min(height, int(np.floor(y1))))
            x2_i = max(0, min(width, int(np.ceil(x2))))
            y2_i = max(0, min(height, int(np.ceil(y2))))
            if x2_i > x1_i and y2_i > y1_i:
                mask[y1_i:y2_i, x1_i:x2_i] = True
        return mask

    def segment_boxes(self, image: Image.Image, boxes: list[list[float]]) -> RingMoSamResult:
        original_size = image.size
        class_map = self.predict_class_map(image)
        if class_map.shape != (original_size[1], original_size[0]):
            class_map_image = Image.fromarray(class_map, mode="L").resize(original_size, Image.Resampling.NEAREST)
            class_map = np.asarray(class_map_image, dtype=np.uint8)

        class_mask = np.isin(class_map, self.class_ids)
        box_mask = self.boxes_to_mask(boxes, class_map.shape)
        merged_mask = class_mask & box_mask
        return RingMoSamResult(
            class_map=class_map,
            merged_mask=merged_mask,
            input_boxes=boxes,
            class_ids=self.class_ids,
        )
