"""
Training Script for SIU Model

This script:
1. Loads correct structure data (original annotations)
2. Generates incorrect structure data using synthesis methods
3. Extracts geometric features from both datasets
4. Trains a binary classifier to distinguish correct vs incorrect
5. Optimizes the instance score threshold
6. Saves the trained model

Reference: Section 3 of the SIU paper
"""

import os
import numpy as np
import joblib
from typing import Dict, Any, Tuple, List
import logging
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils import (
    load_config, setup_logging, create_output_dirs,
    load_yolo_annotation, get_dataset_files, get_timestamp
)
from .feature_engineering import extract_geometric_features
from .data_synthesis import generate_synthetic_dataset

logger = logging.getLogger('SIU.Train')


def load_annotations_from_dataset(
    images_dir: str,
    labels_dir: str,
    img_width: int = 640,
    img_height: int = 640,
    max_samples: int = None
) -> List[List[Dict[str, Any]]]:
    """
    Load all annotations from dataset

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO labels
        img_width: Image width
        img_height: Image height
        max_samples: Maximum number of samples to load (None = all)

    Returns:
        List of annotation sets (each is a list of boxes for one image)
    """
    dataset_files = get_dataset_files(images_dir, labels_dir)

    if max_samples:
        dataset_files = dataset_files[:max_samples]

    annotations_list = []

    logger.info(f"Loading {len(dataset_files)} annotation files from {labels_dir}")

    for img_path, label_path in tqdm(dataset_files, desc="Loading annotations"):
        boxes = load_yolo_annotation(label_path, img_width, img_height)
        if boxes:  # Only include images with at least one object
            annotations_list.append(boxes)

    logger.info(f"Loaded {len(annotations_list)} valid annotation sets")
    return annotations_list


def prepare_training_data(
    correct_samples: List[List[Dict[str, Any]]],
    incorrect_samples: List[List[Dict[str, Any]]],
    config: Dict[str, Any],
    img_width: int = 640,
    img_height: int = 640
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare feature matrix and labels for training

    Args:
        correct_samples: List of correct annotation sets
        incorrect_samples: List of incorrect annotation sets
        config: Configuration dictionary
        img_width: Image width
        img_height: Image height

    Returns:
        Tuple of (X, y) where X is feature matrix and y is labels
    """
    features_list = []
    labels_list = []

    logger.info("Extracting features from correct samples...")
    for annotations in tqdm(correct_samples, desc="Correct samples"):
        try:
            features = extract_geometric_features(
                annotations, img_width, img_height, config
            )
            if len(features) > 0:
                features_list.append(features)
                labels_list.append(1)  # 1 = correct structure
        except Exception as e:
            logger.warning(f"Error extracting features from correct sample: {e}")
            continue

    logger.info("Extracting features from incorrect samples...")
    for annotations in tqdm(incorrect_samples, desc="Incorrect samples"):
        try:
            features = extract_geometric_features(
                annotations, img_width, img_height, config
            )
            if len(features) > 0:
                features_list.append(features)
                labels_list.append(0)  # 0 = incorrect structure
        except Exception as e:
            logger.warning(f"Error extracting features from incorrect sample: {e}")
            continue

    # Pad features to same length (handle variable number of objects)
    if features_list:
        max_length = max(len(f) for f in features_list)
        logger.info(f"Max feature length: {max_length}")

        padded_features = []
        for features in features_list:
            if len(features) < max_length:
                padding = np.zeros(max_length - len(features))
                features = np.concatenate([features, padding])
            elif len(features) > max_length:
                features = features[:max_length]
            padded_features.append(features)

        X = np.array(padded_features)
        y = np.array(labels_list)

        logger.info(f"Prepared dataset: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"Class distribution: Correct={np.sum(y==1)}, Incorrect={np.sum(y==0)}")

        return X, y
    else:
        raise ValueError("No valid features extracted from dataset")


def train_siu_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[Any, StandardScaler]:
    """
    Train the SIU classification model

    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration dictionary

    Returns:
        Tuple of (trained model, feature scaler)
    """
    model_config = config.get('siu_model', {})
    classifier_type = model_config.get('classifier_type', 'GradientBoosting')

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train classifier
    if classifier_type == 'GradientBoosting':
        params = model_config.get('gradient_boosting', {})
        model = GradientBoostingClassifier(
            n_estimators=params.get('n_estimators', 200),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 5),
            min_samples_split=params.get('min_samples_split', 10),
            min_samples_leaf=params.get('min_samples_leaf', 5),
            subsample=params.get('subsample', 0.8),
            random_state=params.get('random_state', 42),
            verbose=1
        )
    elif classifier_type == 'RandomForest':
        params = model_config.get('random_forest', {})
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 200),
            max_depth=params.get('max_depth', 15),
            min_samples_split=params.get('min_samples_split', 10),
            min_samples_leaf=params.get('min_samples_leaf', 5),
            random_state=params.get('random_state', 42),
            verbose=1,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    logger.info(f"Training {classifier_type} classifier...")
    model.fit(X_train_scaled, y_train)

    # Cross-validation score
    cv_folds = model_config.get('cv_folds', 5)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds, scoring='f1')
    logger.info(f"Cross-validation F1 scores: {cv_scores}")
    logger.info(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model, scaler


def optimize_instance_threshold(
    model: Any,
    scaler: StandardScaler,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any]
) -> float:
    """
    Optimize the instance score threshold for classification

    The instance score is calculated as: N_true / (N_true + N_false)
    where N_true = number of correctly validated parts
    and N_false = number of incorrectly validated parts

    Args:
        model: Trained classifier
        scaler: Fitted feature scaler
        X_val: Validation features
        y_val: Validation labels
        config: Configuration dictionary

    Returns:
        Optimal threshold value
    """
    X_val_scaled = scaler.transform(X_val)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

    # Try different thresholds
    thresholds = np.arange(0.5, 1.0, 0.05)
    best_threshold = 0.85
    best_f1 = 0.0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    logger.info(f"Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold


def evaluate_model(
    model: Any,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    output_dir: str
) -> Dict[str, float]:
    """
    Evaluate model performance on test set

    Args:
        model: Trained classifier
        scaler: Fitted feature scaler
        X_test: Test features
        y_test: Test labels
        threshold: Classification threshold
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary of evaluation metrics
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Incorrect', 'Correct'])

    # Log results
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{cm}")
    logger.info("\nClassification Report:")
    logger.info(f"\n{report}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Incorrect', 'Correct'])
    plt.yticks(tick_marks, ['Incorrect', 'Correct'])

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {cm_path}")
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    logger.info(f"ROC curve saved to {roc_path}")
    plt.close()

    return metrics


def main_train(config_path: str = "config/config.yaml") -> None:
    """
    Main training pipeline

    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    logger_instance = setup_logging(config)
    create_output_dirs(config)

    logger.info("=" * 60)
    logger.info("SIU MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # Dataset configuration
    dataset_config = config.get('dataset', {})
    img_width = dataset_config.get('image_size', 640)
    img_height = dataset_config.get('image_size', 640)

    train_images = dataset_config.get('train_images')
    train_labels = dataset_config.get('train_labels')

    num_classes = len(config.get('classes', []))

    # Load annotations
    logger.info("Step 1: Loading annotations...")
    annotations_list = load_annotations_from_dataset(
        train_images,
        train_labels,
        img_width,
        img_height,
        max_samples=config.get('training', {}).get('max_samples_per_class')
    )

    # Generate synthetic incorrect data
    logger.info("Step 2: Generating synthetic incorrect data...")
    correct_samples, incorrect_samples = generate_synthetic_dataset(
        annotations_list,
        config,
        img_width,
        img_height,
        num_classes
    )

    # Prepare training data
    logger.info("Step 3: Preparing training data...")
    X, y = prepare_training_data(
        correct_samples,
        incorrect_samples,
        config,
        img_width,
        img_height
    )

    # Split data
    training_config = config.get('training', {})
    test_size = training_config.get('test_size', 0.2)
    random_state = training_config.get('random_state', 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train model
    logger.info("Step 4: Training SIU model...")
    model, scaler = train_siu_model(X_train, y_train, config)

    # Optimize threshold
    logger.info("Step 5: Optimizing instance score threshold...")
    threshold = optimize_instance_threshold(model, scaler, X_test, y_test, config)

    # Evaluate model
    logger.info("Step 6: Evaluating model...")
    output_dir = config.get('output', {}).get('results_dir', 'outputs')
    metrics = evaluate_model(model, scaler, X_test, y_test, threshold, output_dir)

    # Save model
    logger.info("Step 7: Saving model...")
    models_dir = config.get('output', {}).get('models_dir', 'models')
    os.makedirs(models_dir, exist_ok=True)

    timestamp = get_timestamp()
    model_path = os.path.join(models_dir, f'siu_model_{timestamp}.pkl')
    scaler_path = os.path.join(models_dir, f'scaler_{timestamp}.pkl')
    threshold_path = os.path.join(models_dir, f'threshold_{timestamp}.txt')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(threshold_path, 'w') as f:
        f.write(str(threshold))

    # Also save as "latest" for easy inference
    joblib.dump(model, os.path.join(models_dir, 'siu_model_latest.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler_latest.pkl'))
    with open(os.path.join(models_dir, 'threshold_latest.txt'), 'w') as f:
        f.write(str(threshold))

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Threshold saved to {threshold_path}")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main_train()
