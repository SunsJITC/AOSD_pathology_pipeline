# AOSD Histopathology Survival Pipeline

This repository implements a WSI-based histopathology pipeline for AOSD risk modeling, survival analysis, attention heatmap generation, cross-encoder consensus patch export, and Figure 5 panel preparation.

The main workflow is:

1. Extract tile-level features from WSIs
2. Aggregate multiple slides into patient-level bags
3. Train a MIL-Cox survival model


---

## 1. Repository structure

```text
aosd_prognosis_pipeline/
├── models/
│   ├── encoders.py               # UNI2-h / Virchow2 / DINOv2 / Prov-GigaPath
│   └── mil.py                    # Gated-attention MIL
├── utils/
│   ├── io.py
│   └── metrics.py                # Time-ROC, KM, C-index helpers
├── wsi/
│   ├── tiling.py                 # WSI loading, tiling, thumbnails
│   ├── stain_norm.py             # Macenko normalization
│   └── heatmap.py                # Standard heatmap rendering
├── extract_feats.py              # Tile feature extraction
├── build_patient_bags.py         # Patient-level bag aggregation
├── train_mil_cox.py              # Main training script
└── ckpts/                        # Encoder checkpoints
```

---

## 2. Environment

### Install with pip

```bash
pip install -r requirements.txt
```

### Install with conda

```bash
conda env create -f environment.yml
conda activate <env-name>
```

Main dependencies:

- Python 3.10+
- PyTorch >= 2.1
- scikit-learn >= 1.3
- lifelines >= 0.28
- OpenCV >= 4.9
- tifffile
- openslide-python
- timm
- transformers

---

## 3. Configuration files


There are `base_clean_*.yaml` files for more stable encoder switching and workdir control.

Important config fields:

- `run.workdir`: output directory for the current encoder/run
- `data.wsi_glob`: WSI search paths
- `data.clinic_xlsx`: clinical table path
- `encoder.kind`: encoder type
- `encoder.weights_path`: checkpoint path
- `encoder.input_size`: encoder input size
- `encoder.embed_dim`: feature dimension
- `tiling.patch_size`: original tile size
- `tiling.read_level`: WSI read level
- `tiling.thumbnail_size`: maximum thumbnail size

### Current default experimental settings

- Original tile size: `256 × 256`
- Encoder input size: `224 × 224`
- Batch size: `16`
- Normalization:
  - `mean = [0.485, 0.456, 0.406]`
  - `std = [0.229, 0.224, 0.225]`

---

## 4. Data flow

### Step 1. Extract tile features from WSIs

```bash
python extract_feats.py --config configs/base_uni2.yaml
```

Outputs:

```text
{workdir}/feats/*.npz
{workdir}/thumbs/*.png
{workdir}/logs/extract.log
```

Each `npz` typically contains:

- `feats`: tile feature matrix
- `coords`: tile top-left coordinates
- `tile_size`
- `thumb_scale`

### Step 2. Build patient-level bags

```bash
python build_patient_bags.py --config configs/base_uni2.yaml
```

Outputs:

```text
{workdir}/patient_bags/*.npz
{workdir}/bag_size_report.csv
```

Multiple slides belonging to the same patient are aggregated at this stage.

### Step 3. Train the MIL-Cox model

```bash
python train_mil_cox.py --config configs/base_uni2.yaml
```

Typical outputs:

```text
{workdir}/models/fold1_mil.pt ... fold5_mil.pt
{workdir}/models/fold1_head.pt ... fold5_head.pt
{workdir}/models/best_hparams.json
{workdir}/scores.csv
{workdir}/report.txt
{workdir}/timeROC_*.png
```

Current `train_mil_cox.py` characteristics:

