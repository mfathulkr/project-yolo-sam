# Dataset Provenance Audit

- Durum: **GEÇTİ**
- Profil: `samrs_sota`
- Kök: `/home/ssyzai/projects/yolo-sam/project-yolo-sam/datasets/samrs/raw/sota`
- Anotasyon formatı: `samrs_pickle_instances`
- Görüntü: 17555
- Anotasyon dosyası: 17555
- Instance: 615407

## Category Mapping

| Split / ID | Category |
|---:|---|
| 0 | large-vehicle |
| 1 | swimming-pool |
| 2 | helicopter |
| 3 | bridge |
| 4 | plane |
| 5 | ship |
| 6 | soccer-ball-field |
| 7 | basketball-court |
| 8 | ground-track-field |
| 9 | small-vehicle |
| 10 | baseball-diamond |
| 11 | tennis-court |
| 12 | roundabout |
| 13 | storage-tank |
| 14 | harbor |
| 15 | container-crane |
| 16 | airport |
| 17 | helipad |

## Bulgular

### INFO: `AUTHORITATIVE_RDET_VALIDATED`

Every instance label and original RBox/RHBox geometry matches the authoritative rotated-detection annotations.

```json
{
  "files": 17555,
  "instances": 615407,
  "rdet_directory": "/home/ssyzai/projects/yolo-sam/project-yolo-sam/datasets/samrs/raw/sota/trainval/rbbtxts"
}
```

### WARNING: `PICKLE_CATEGORY_STRINGS_IGNORED`

Pickle category-name strings do not describe this SOTA subset. Numeric labels and geometry were exhaustively validated against the authoritative detection annotations, whose category mapping is used.

```json
{
  "pickle_mapping": {
    "0": [
      "A220"
    ],
    "1": [
      "A321"
    ],
    "2": [
      "A330"
    ],
    "3": [
      "A350"
    ],
    "4": [
      "ARJ21"
    ],
    "5": [
      "Baseball-Field"
    ],
    "6": [
      "Basketball-Court"
    ],
    "7": [
      "Boeing737"
    ],
    "8": [
      "Boeing747"
    ],
    "9": [
      "Boeing777"
    ],
    "10": [
      "Boeing787"
    ],
    "11": [
      "Bridge"
    ],
    "12": [
      "Bus"
    ],
    "13": [
      "C919"
    ],
    "14": [
      "Cargo-Truck"
    ],
    "15": [
      "Dry-Cargo-Ship"
    ],
    "16": [
      "Dump-Truck"
    ],
    "17": [
      "Engineering-Ship"
    ]
  },
  "authoritative_mapping": {
    "0": [
      "large-vehicle"
    ],
    "1": [
      "swimming-pool"
    ],
    "2": [
      "helicopter"
    ],
    "3": [
      "bridge"
    ],
    "4": [
      "plane"
    ],
    "5": [
      "ship"
    ],
    "6": [
      "soccer-ball-field"
    ],
    "7": [
      "basketball-court"
    ],
    "8": [
      "ground-track-field"
    ],
    "9": [
      "small-vehicle"
    ],
    "10": [
      "baseball-diamond"
    ],
    "11": [
      "tennis-court"
    ],
    "12": [
      "roundabout"
    ],
    "13": [
      "storage-tank"
    ],
    "14": [
      "harbor"
    ],
    "15": [
      "container-crane"
    ],
    "16": [
      "airport"
    ],
    "17": [
      "helipad"
    ]
  }
}
```

### WARNING: `SOURCE_SCENE_SPLIT_LEAKAGE`

The raw train and validation lists contain tiles from common source scenes. A source-scene-safe resplit is required.

```json
{
  "overlap_count": 535,
  "examples": [
    "P0000",
    "P0011",
    "P0019",
    "P0020",
    "P0021",
    "P0022",
    "P0023",
    "P0036",
    "P0041",
    "P0047",
    "P0063",
    "P0064",
    "P0065",
    "P0066",
    "P0082",
    "P0111",
    "P0113",
    "P0136",
    "P0141",
    "P0167"
  ]
}
```
