# Dataset Provenance Audit

- Durum: **GEÇTİ**
- Profil: `isaid`
- Kök: `/home/ssyzai/projects/yolo-sam/project-yolo-sam/datasets/isaid/raw`
- Anotasyon formatı: `coco_instance_segmentation`
- Görüntü: 1869
- Anotasyon dosyası: 2
- Instance: 471438

## Category Mapping

| Split / ID | Category |
|---:|---|
| train:1 | storage_tank |
| train:10 | Soccer_ball_field |
| train:11 | baseball_diamond |
| train:12 | Bridge |
| train:13 | basketball_court |
| train:14 | Roundabout |
| train:15 | Helicopter |
| train:2 | Large_Vehicle |
| train:3 | Small_Vehicle |
| train:4 | plane |
| train:5 | ship |
| train:6 | Swimming_pool |
| train:7 | Harbor |
| train:8 | tennis_court |
| train:9 | Ground_Track_Field |
| val:1 | ship |
| val:10 | Helicopter |
| val:11 | Swimming_pool |
| val:12 | Roundabout |
| val:13 | Soccer_ball_field |
| val:14 | plane |
| val:15 | Harbor |
| val:2 | storage_tank |
| val:3 | baseball_diamond |
| val:4 | tennis_court |
| val:5 | basketball_court |
| val:6 | Ground_Track_Field |
| val:7 | Bridge |
| val:8 | Large_Vehicle |
| val:9 | Small_Vehicle |

## Bulgular

### WARNING: `CATEGORY_IDS_DIFFER_BY_SPLIT`

The target category uses different numeric IDs across iSAID splits. Category selection must use each split's name-to-ID mapping.

```json
{
  "train": 3,
  "val": 9
}
```
