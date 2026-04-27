# AOSD Histopathology Survival Pipeline

This repository implements a WSI-based histopathology pipeline for AOSD risk modeling, survival analysis, attention heatmap generation, cross-encoder consensus patch export, and Figure 5 panel preparation.

The main workflow is:

1. Extract tile-level features from WSIs
2. Aggregate multiple slides into patient-level bags
3. Train a MIL-Cox survival model
4. Generate WSI-level attention heatmaps
5. Export consensus high/low patches across encoders
6. Prepare publication-ready Figure 5 panel assets

---

## 1. Repository structure

```text
aosd_prognosis_pipeline/
├── configs/                      # Encoder and run configurations
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
├── generate_attention_maps.py    # Standard batch heatmaps
├── generate_attention_maps_v2.py # High-resolution heatmaps / Figure 5 workflow
├── export_consensus_patches.py   # Consensus high/low patches across encoders
├── generate_figure5_panels.py    # Figure 5 panel assets
├── find_nearby_merge_tile.py     # Local low/high tile search in a region
├── rebuild_thumbs.py             # Rebuild thumbnails only
├── make_colorbar.py              # Standalone colorbar generation
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

Common configs are stored in `configs/`:

- `base_uni2.yaml`
- `base_virchow2.yaml`
- `base_provgigapath.yaml`
- `base_dinov2.yaml`

There are also `base_clean_*.yaml` files for more stable encoder switching and workdir control.

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

- gated-attention MIL
- final risk is produced directly by `risk_head(z)`
- 5-fold training inside the training pool
- supports downstream attention heatmaps and Figure 5 workflow

---

## 5. Standard attention heatmaps

To generate batch WSI heatmaps:

```bash
python generate_attention_maps.py --config configs/base_uni2.yaml
```

Outputs:

```text
{workdir}/attention_maps/
├── heatmaps/          # with colorbar
├── heatmaps_raw/      # without colorbar
├── top_tiles/         # top-K tile JSON files
└── attention_summary.csv
```

By default, the script processes:

- `train`
- `val`
- `external`

This can be overridden with `--splits`.

### Standard heatmap parameters

- default `top_k = 20`
- attention is averaged across 5 fold models
- standard rendering uses soft-kernel smoothing from `wsi/heatmap.py`

---

## 6. High-resolution heatmaps and Figure 5

For high-resolution visualization, cross-encoder agreement, or Figure 5 preparation, use:

```bash
python generate_attention_maps_v2.py --config configs/base_clean_uni2.yaml --encoder_name UNI2-h
```

Typical use cases:

- high-resolution heatmaps for selected slides
- slide-specific rendering via `--slide_ids`
- merging multiple encoder results
- generating Figure 5-ready intermediate files

### Process selected WSIs only

```bash
python generate_attention_maps_v2.py \
  --config configs/base_clean_uni2.yaml \
  --encoder_name UNI2-h \
  --slide_ids P202501080-1 P202500507-1
```

### Merge multiple encoders

```bash
python generate_attention_maps_v2.py \
  --merge \
  --config configs/base_clean_uni2.yaml \
  --encoder_dirs \
    /path/to/UNI2/attention_maps \
    /path/to/Virchow2/attention_maps \
    /path/to/provgigapath/attention_maps \
  --encoder_names UNI2-h Virchow2 provgigapath \
  --patient_ids P202501080 P202500507 \
  --output_dir /path/to/Figure5_output
```

---

## 7. Consensus patches across encoders

To export consensus high/low patches from three encoder attention maps:

```bash
python export_consensus_patches.py \
  --config configs/base_clean_uni2.yaml \
  --encoder_dirs \
    /path/to/UNI2/attention_maps \
    /path/to/Virchow2/attention_maps \
    /path/to/provgigapath/attention_maps \
  --encoder_names UNI2-h Virchow2 provgigapath \
  --patient_ids P202501080 P202500507 \
  --output_dir ./consensus_patches_out \
  --top_pct 0.15 \
  --bottom_pct 0.15 \
  --patch_size 512 \
  --min_encoders 3
```

Output structure:

```text
consensus_patches_out/{patient_id}/
├── consensus_high/
├── consensus_low/
└── consensus_patches.csv
```

Notes:

- `rank 1` in the low group means the lowest mean attention
- `top_pct` / `bottom_pct` define candidate cutoffs
- `min_encoders` controls how many encoders must agree

---

## 8. Figure 5 panel assets

After manually selecting consensus ranks, generate panel components:

```bash
python generate_figure5_panels.py \
  --config configs/base_clean_uni2.yaml \
  --encoder_dirs \
    /path/to/UNI2/attention_maps \
    /path/to/Virchow2/attention_maps \
    /path/to/provgigapath/attention_maps \
  --encoder_names UNI2-h Virchow2 provgigapath \
  --patient_id P202501080 \
  --consensus_csv /path/to/consensus_patches.csv \
  --high_ranks 3 5 6 7 \
  --low_ranks 19 22 26 29 30 \
  --output_dir ./figure5_panels/P202501080
```

Outputs include:

- `{patient}_wsi_annotated.tiff`
- `{patient}_wsi_annotated_numbered.tiff`
- `{patient}_wsi_annotated_rot45.tiff`
- `{patient}_wsi_annotated_rot45_numbered.tiff`
- `{patient}_heatmap_{encoder}.tiff`
- `{patient}_patch_01.tiff`
- `{patient}_patch_01_rot45.tiff`
- `{patient}_panel_info.json`

Extra options:

- `--extra_low_tile_indices`
- `--extra_high_tile_indices`
- `--extra_low_coords`
- `--extra_high_coords`

These are useful when manually adding a new tile after consensus review.

---

## 9. Search for nearby replacement tiles

If the exported consensus patches are not satisfactory, search again within a local region:

```bash
python find_nearby_merge_tile.py \
  --patient_id P202500507 \
  --config configs/base_clean_uni2.yaml \
  --consensus_csv /path/to/consensus_patches.csv \
  --encoder_dirs \
    /path/to/UNI2/attention_maps \
    /path/to/Virchow2/attention_maps \
    /path/to/provgigapath/attention_maps \
  --encoder_names UNI2-h Virchow2 provgigapath \
  --anchor_coords 9216 8960 34560 13056 34560 12544 \
  --search_group low \
  --padding_tiles 1.0 \
  --topk 20 \
  --patch_size 512 \
  --output_csv ./new_low_merge_candidates.csv
```

Outputs:

- candidate CSV
- candidate patch images

---

## 10. Rebuild thumbnails only

If features already exist but thumbnails need to be rebuilt or aligned:

```bash
python rebuild_thumbs.py --config configs/base_uni2.yaml --overwrite
```

Then regenerate the heatmaps.

---

## 11. Standalone color bar

To generate a standalone attention color bar:

```bash
python make_colorbar.py
```

Default output:

```text
./colorbar.tiff
```

---

## 12. Common issues

### 1. Heatmap does not align with the WSI

Check:

- whether `thumbs/` are up to date
- whether `thumb_scale` matches the current thumbnails
- whether the updated `generate_attention_maps.py` / `generate_attention_maps_v2.py` is being used

### 2. No result for a given slide in `attention_maps/`

Check:

- whether `feats/{slide_id}.npz` exists
- whether `thumbs/{slide_id}.png` exists
- whether the correct `--slide_ids` were passed

### 3. Patient aggregation across multiple slides

Default behavior:

- true slide suffixes such as `P20211339-1` or `1234567-2` are stripped
- clinical IDs containing hyphens, such as `22-1983`, are preserved

### 4. Heatmap edges are too sharp or too blocky

- for standard batch heatmaps, prefer `generate_attention_maps.py`
- for Figure 5, use `generate_attention_maps_v2.py`
- if needed, increase thumbnail resolution before tuning local ROI visualization

---

## 13. Recommended execution order

For a full run from raw WSIs:

```text
extract_feats.py
→ build_patient_bags.py
→ train_mil_cox.py
→ generate_attention_maps.py
→ export_consensus_patches.py
→ generate_figure5_panels.py
```

If the model is already trained and you only need visualization:

```text
generate_attention_maps_v2.py
→ export_consensus_patches.py
→ find_nearby_merge_tile.py (optional)
→ generate_figure5_panels.py
```

---

## 14. Notes

- `train_mil_cox.py` is the current main training script
- `train_mil_cox-1.py` is an older preserved version and is not recommended as the main workflow
- `base.yaml` is often manually switched between encoders; for reproducibility, `base_clean_*.yaml` is usually safer

