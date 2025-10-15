# Structured Instance Understanding with Boundary Box Relationships in Object Detection System

[![Paper](https://img.shields.io/badge/Paper-ACM%20Digital%20Library-blue)](https://doi.org/10.1145/3643487.3662729)
[![Dataset](https://img.shields.io/badge/Dataset-Roboflow-brightgreen)](https://universe.roboflow.com/project-p5nyc/car-parts-o7dlr)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

Implementation of the SIU (Structured Instance Understanding) approach for validating object detection results by analysing geometric relationships between detected bounding boxes. The project accompanies the publication **“Structured Instance Understanding with Boundary Box Relationships in Object Detection System”** and provides an end‑to‑end pipeline covering training, inference, evaluation, and visualization.

## Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Train the SIU classifier](#train-the-siu-classifier)
  - [Single-image inference](#single-image-inference)
  - [Batch inference](#batch-inference)
- [Configuration](#configuration)
- [Methodology](#methodology)
  - [Data synthesis](#data-synthesis)
  - [Feature engineering](#feature-engineering)
  - [Structure classification](#structure-classification)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

Standard detectors such as YOLO frequently return predictions that are spatially inconsistent, duplicated, or missing critical components. SIU mitigates these errors by:

1. Detecting candidate objects with a YOLO backbone.
2. Generating synthetic error cases to model structural failures.
3. Extracting pairwise and global geometric descriptors.
4. Classifying the predicted layout as structurally correct or incorrect.
5. Producing an instance score and optional visualizations for downstream automation.

## Key Capabilities

- Four complementary synthesis strategies (class transform, random add/delete, box shift) for generating realistic negative samples.
- Rich geometric feature set capturing size, distance, angular, overlap, and class relationships.
- Gradient Boosting or Random Forest classifier with automatic threshold optimisation.
- Command line interface supporting training, single-image inference, and batch processing.
- Visualization utilities to annotate detections with validation scores.

## Architecture

```
┌─────────────┐     ┌────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│   Dataset   │ ──► │ Data Synthesis │ ──► │ Feature Engineering │ ──► │ SIU Classifier   │
└─────────────┘     └────────────────┘     └─────────────────────┘     └──────────────────┘
        │                      │                          │                     │
        ▼                      ▼                          ▼                     ▼
   YOLO Detector ─────────────────────────────────────────────────────────► Validation & Reporting
```

## Repository Layout

```
siu-object-detection-validator/
├── config/                  # Configuration files (YAML)
├── src/                     # Core library
│   ├── data_synthesis.py    # Error generation strategies
│   ├── feature_engineering.py
│   ├── inference.py
│   ├── train.py
│   └── utils.py
├── main.py                  # CLI entry point
├── QUICKSTART.md            # Hands-on setup guide
├── IMPLEMENTATION_SUMMARY.md
├── CONTRIBUTE.md
└── ...
```

## Prerequisites

- Python 3.9 or newer.
- CUDA-capable GPU recommended for YOLO training/inference (CPU mode supported).
- At least 8 GB RAM and 5 GB free disk space for dataset and models.

## Installation

```bash
git clone <repository-url>
cd siu-object-detection-validator
pip install -r requirements.txt
```

## Dataset

The project targets the [Car Parts dataset](https://universe.roboflow.com/project-p5nyc/car-parts-o7dlr) (YOLOv11 format, CC BY 4.0 licence).

1. Download the dataset in YOLO v11 format.
2. Extract it to the repository root as `Car parts.v1i.yolov11/`.
3. Confirm the expected structure:

```bash
ls "Car parts.v1i.yolov11"
# train/  valid/  test/  data.yaml
```

Update `config/config.yaml` if your dataset lives elsewhere or if you use a different YOLO dataset.

## Usage

### Train the SIU classifier

```bash
python main.py train --config config/config.yaml
```

The training routine performs data synthesis, feature extraction, model fitting, threshold optimisation, evaluation (confusion matrix, ROC curve), and persists the latest artefacts in `models/` and `outputs/`.

### Single-image inference

```bash
python main.py inference "Car parts.v1i.yolov11/test/images/example.jpg"
```

The command generates:

- Console summary with detection counts and instance score.
- Optional visualization stored under `outputs/visualizations/`.

### Batch inference

```bash
python main.py batch "Car parts.v1i.yolov11/test/images/" --output outputs
```

Processes every supported image in the directory, summarising success/failure counts and aggregated metrics.

## Configuration

`config/config.yaml` controls datasets, YOLO settings, synthesis parameters, feature toggles, learning algorithm, and logging preferences. Common adjustments include:

- **Dataset paths**: `dataset.train_images`, `dataset.train_labels`, etc.
- **YOLO model**: swap `yolo.model_name` for custom weights.
- **Synthesis controls**: tune probability, shift ranges, and ratios per method.
- **Classifier hyperparameters**: select `GradientBoosting` or `RandomForest` and refine estimator settings.
- **Output & logging**: change directories, enable/disable visualizations, or redirect log files.

## Methodology

### Data synthesis

- **Class transform**: randomly reassigns class labels (default probability 0.3).
- **Random add**: inserts spurious boxes either by duplicating existing detections or sampling random shapes.
- **Random delete**: removes essential components while ensuring at least one object remains.
- **Box shift**: displaces boxes by configurable offsets while retaining their original class.

These procedures mirror frequent detector failure cases and balance the binary training dataset.

### Feature engineering

`src/feature_engineering.py` derives a fixed-length descriptor comprising:

- Normalised bounding box dimensions, area, and aspect ratios.
- Pairwise distances, angles (with sine/cosine encoding), and relative offsets.
- Size ratios and log-scaled height/width comparisons.
- IoU overlap statistics and class relationship indicators.
- Global aggregates (object count, central tendency, dispersion, class diversity).

### Structure classification

- Features are scaled with `StandardScaler`.
- A tree-based classifier (Gradient Boosting by default) is trained with cross-validation to maximise F1.
- Post-training, the decision threshold is calibrated over a validation split.
- Inference reuses the persisted scaler, classifier, and threshold, ensuring consistent feature padding/truncation.

## Evaluation

Representative performance on the Car Parts dataset:

| Metric      | Value    |
|-------------|----------|
| Accuracy    | 85 – 95% |
| Precision   | 85 – 92% |
| Recall      | 88 – 95% |
| F1 score    | 85 – 92% |
| ROC AUC     | 0.90 – 0.95 |

The SIU validator typically reduces structural errors by 60 – 80% with <1 s additional processing time per image (hardware dependent). Visualization examples are available under `outputs/visualizations/` once inference has been executed.

## Troubleshooting

- **`No module named 'ultralytics'`** – Install the package: `pip install ultralytics`.
- **CUDA out of memory** – Switch to a smaller YOLO model (`yolov8n.pt`), set `yolo.device: "cpu"`, or lower batch size.
- **Insufficient training data** – Increase `synthesis.synthesis_ratio`, collect more images, or adjust `training.max_samples_per_class`.
- **Low validation accuracy** – Revisit synthesis probabilities, enable additional feature groups, or retune classifier hyperparameters.
- **Custom log path fails** – Ensure `logging.file` includes a directory or relies on the default `logs/siu.log`.

## Resources

- [Quick Start](QUICKSTART.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- ACM Digital Library entry for the paper: [10.1145/3643487.3662729](https://doi.org/10.1145/3643487.3662729)
- Dataset on Roboflow: [Car Parts](https://universe.roboflow.com/project-p5nyc/car-parts-o7dlr)
- YOLOv8 by Ultralytics: <https://github.com/ultralytics/ultralytics>

## Contributing

Contributions are welcome. Please review [CONTRIBUTE.md](CONTRIBUTE.md) for coding standards, testing expectations, and pull request guidelines.

## License

The source code is distributed under the [MIT License](LICENSE). The Car Parts dataset follows the CC BY 4.0 licence; consult the dataset page for permissible usage.

## Contact

- **Research inquiries:** Refer to the published paper or contact the authors through the ACM Digital Library.
- **Implementation questions or issues:** open a GitHub issue.
- **General enquiries:** `ksupasate@gmail.com`.
