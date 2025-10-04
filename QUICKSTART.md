# Quick Start Guide

Get up and running with SIU Object Detection Validator in 5 minutes!

## 📦 Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- ultralytics (YOLO)
- scikit-learn (Classifier)
- opencv-python (Image processing)
- And other dependencies

## 🏋️ Step 2: Train the SIU Model

```bash
python main.py train
```

**What happens:**
1. Loads 2,796 training images from the dataset
2. Generates synthetic incorrect structures using 4 methods
3. Extracts geometric features (distance, angle, size ratios, etc.)
4. Trains GradientBoosting classifier
5. Optimizes threshold on validation set
6. Saves model to `models/siu_model_latest.pkl`

**Expected time:** 5-15 minutes (depending on hardware)

**Expected output:**
```
Training SIU model...
Cross-validation F1 scores: [0.89, 0.91, 0.88, 0.90, 0.89]
Mean CV F1: 0.894 (+/- 0.011)
Optimal threshold: 0.85 (F1=0.9123)

MODEL EVALUATION RESULTS
Accuracy:  0.9234
Precision: 0.9156
Recall:    0.9312
F1 Score:  0.9233

✓ Training completed successfully!
```

## 🔍 Step 3: Test on a Single Image

```bash
python main.py inference "Car parts.v1i.yolov11/test/images/resized_honda_cars-141-_jpg.rf.2759ceabacbca927c9d7112c0b762818.jpg"
```

**Output:**
```
INFERENCE RESULTS
Detected objects: 7
Instance score: 0.9234
Structure valid: ✓ YES

Detected objects:
  1. Rear bumper: confidence=0.923
  2. Car boot: confidence=0.887
  3. Rear light - -L-: confidence=0.945
  4. Rear light - -R-: confidence=0.941
  ...

Visualization saved: outputs/visualizations/resized_honda_cars-141_validated.jpg
```

## 📊 Step 4: Batch Process Test Set

```bash
python main.py batch "Car parts.v1i.yolov11/test/images/" --output results/
```

This processes all 167 test images and generates:
- Individual visualizations
- Summary statistics
- Instance scores for each image

## 🎯 What's Next?

### Improve Performance

1. **Tune hyperparameters** in `config/config.yaml`:
   - Adjust synthesis parameters
   - Enable/disable features
   - Change classifier settings

2. **Use a larger YOLO model**:
   ```yaml
   yolo:
     model_name: "yolov8m.pt"  # medium model
   ```

3. **Train on more data**:
   - Add more training samples
   - Increase synthesis ratio

### Understand Your Results

**High instance score (>0.85)**: Structure looks correct
- All expected parts present
- Reasonable spatial relationships
- No obvious duplicates or misplacements

**Low instance score (<0.85)**: Structure might be incorrect
- Missing parts
- Duplicate detections
- Parts in wrong positions
- Wrong class labels

### Integration

Use the SIU validator in your own pipeline:

```python
from src.inference import SIUValidator
from src.utils import load_config

config = load_config('config/config.yaml')

validator = SIUValidator(
    yolo_model_path='yolov8n.pt',
    siu_model_path='models/siu_model_latest.pkl',
    scaler_path='models/scaler_latest.pkl',
    threshold_path='models/threshold_latest.txt',
    config=config
)

# Run detection
detections = validator.run_yolo_detection('image.jpg')

# Validate structure
validation = validator.validate_structure(detections, 640, 640)

if validation['is_correct_structure']:
    print("✓ Structure is correct!")
else:
    print("✗ Structure validation failed!")
```

## 🐛 Common Issues

### Issue: Import errors

```bash
pip install --upgrade -r requirements.txt
```

### Issue: YOLO model not found

First run will download YOLOv8 automatically. Wait for download to complete.

### Issue: Out of memory

Reduce number of training samples:
```yaml
training:
  max_samples_per_class: 500  # Use fewer samples
```

## 📈 Expected Results

After training on the Car Parts dataset:

| Metric | Expected Range |
|--------|---------------|
| Accuracy | 85-95% |
| Precision | 85-92% |
| Recall | 88-95% |
| F1 Score | 85-92% |
| Optimal Threshold | 0.80-0.90 |

## 🎓 Learn More

- Read the full [README.md](README.md) for detailed documentation
- Check the research paper for methodology details
- Explore `config/config.yaml` for all configuration options
- Review code in `src/` directory for implementation details

---

Happy validating! 🚀
