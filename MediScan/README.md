# MediScan — Medical Image Classification

Educational starter project based on the Code A Nova MediScan 15-day brief.

## Features
- ResNet50 transfer learning (optional EfficientNet-B0)
- Train/validation/test ImageFolder data pipeline
- Resize, ImageNet normalization, flips, rotations, color jitter
- CrossEntropy loss, Adam, learning-rate scheduler
- Accuracy, per-class precision/recall/F1, confusion matrix, ROC-AUC where calculable
- Grad-CAM overlay and Streamlit upload interface
- Model card template and basic test

> Educational prototype only. Not clinically validated and must not be used for diagnosis, triage, or treatment. Confidence scores are not clinical certainty.

## Install
Python 3.10+ (brief requests 3.11+):
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Install the PyTorch build appropriate for your CPU/CUDA setup from the official PyTorch selector.

## Dataset layout
Put a publicly available, properly licensed dataset in:
```
data/
  train/class_a/*.jpg
  train/class_b/*.jpg
  val/class_a/*.jpg
  val/class_b/*.jpg
  test/class_a/*.jpg
  test/class_b/*.jpg
```
Use the same class folder names in every split. Split by patient/person where possible to avoid leakage. Dataset and trained weights are not included. First-time pretrained ResNet50 setup may download ImageNet weights; use `--pretrained false` if unavailable (usually lower performance).

## Train
```bash
python train.py --data-dir data --epochs 10 --batch-size 16
```
Options: `--model resnet50|efficientnet_b0`, `--pretrained true|false`.
Best checkpoint: `checkpoints/best_model.pt`.

## Evaluate
```bash
python evaluate.py --data-dir data --checkpoint checkpoints/best_model.pt
```
Writes `reports/test_metrics.json`.

## Web app
```bash
streamlit run app.py
```
Upload a JPEG/PNG image to view predicted class, model score, and Grad-CAM. Score is not a calibrated clinical probability.

## Tests
```bash
pytest -q
```

## Brief mapping
Days 1–4: dataset preparation and DataLoaders. Days 5–8: transfer learning, training and tuning. Days 9–10: metrics and Grad-CAM. Day 11: second-architecture comparison (extension). Day 12: Streamlit UI. Day 13: MLflow/W&B tracking (extension). Day 14: error/bias analysis. Day 15: deployment, model card and presentation.

The brief's >85% accuracy is a target, not a guarantee. Report actual held-out test performance. Advanced features such as DICOM conversion, batch prediction, CSV export, hosted deployment and experiment tracking need further implementation.
