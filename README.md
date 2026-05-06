# Corrected HRNet 300W code, v2

## Main architectural changes

1. `config.py` is now the single source of truth for paths and experiment settings.
   - Dataset path
   - Checkpoint paths
   - Output directories
   - Training hyperparameters
   - Evaluation thresholds
   - NME normalization mode
   - Alignment template and output size

2. `DeepLake300W.__getitem__()` now returns three values:

```python
img, heatmaps, pts
```

`pts` is the true landmark coordinate after crop and resize. Evaluation should use `pts` directly as GT.

3. `evaluate.py` no longer decodes GT points from GT heatmaps.

Old behavior:

```python
gt_pts = heatmaps_to_pts(gt_hm, IMG_SIZE)
```

Corrected behavior:

```python
for imgs, _gt_hm, gt_pts in loader:
    ...
```

4. `evaluate.py` now reports:
   - Mean NME
   - Median NME
   - Failure rate
   - AUC
   - Per-region pixel error
   - Per-region normalized error

5. `predict_align.py` now reads real GT landmarks directly and adds quantitative alignment metrics.

## Important config entries

Change these in `config.py` instead of editing scripts:

```python
EVAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "hrnet_epoch_200.pth"
EVAL_OUT_DIR = TEST_DIR / "full_eval_epoch_200_fixed"
NME_NORM_TYPE = "inter_ocular"
FAILURE_THRESHOLD = 0.08
AUC_THRESHOLD = 0.08

ALIGN_CHECKPOINT_PATH = EVAL_CHECKPOINT_PATH
ALIGN_OUT_DIR = TEST_DIR / "single_predict_fixed"
ALIGN_OUTPUT_SIZE = 112
```

## How to run

```bash
python train.py
python evaluate.py
python predict_align.py
```

If another script still has:

```python
img, heatmaps = dataset[index]
```

change it to:

```python
img, heatmaps, pts = dataset[index]
```

or:

```python
img, heatmaps, _ = dataset[index]
```
