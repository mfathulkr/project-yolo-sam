from __future__ import annotations

from typing import Protocol

from yolo_sam.data.contracts import (
    BBoxXYWH,
    DatasetIdentity,
    ImageRecord,
    InstanceRecord,
    MaskReference,
    ReferenceType,
)


class DatasetAdapter(Protocol):
    def inspect_identity(self) -> DatasetIdentity: ...

    def list_source_scenes(self) -> list[str]: ...

    def list_images(self) -> list[ImageRecord]: ...

    def list_instances(self, image_id: str) -> list[InstanceRecord]: ...

    def get_original_bbox(self, instance_id: str) -> BBoxXYWH: ...

    def get_reference_mask(
        self,
        instance_id: str,
        reference_type: ReferenceType,
    ) -> MaskReference: ...
