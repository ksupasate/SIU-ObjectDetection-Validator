"""
Inference Pipeline for SIU Object Detection Validator

End-to-end pipeline that:
1. Loads pre-trained YOLO model for object detection
2. Loads trained SIU classifier for structural validation
3. Runs detection on input image
4. Validates structural correctness using geometric features
5. Returns validated results with instance score

Reference: Section 3.2 of the SIU paper
"""

import os
import numpy as np
import cv2
import joblib
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics not installed. YOLO functionality will be limited.")

from .utils import (
    load_config, setup_logging, visualize_boxes, get_timestamp
)
from .feature_engineering import extract_geometric_features

logger = logging.getLogger('SIU.Inference')


class SIUValidator:
    """
    SIU Validator for Object Detection

    Validates the structural correctness of object detection results
    using geometric relationship analysis.
    """

    def __init__(
        self,
        yolo_model_path: str,
        siu_model_path: str,
        scaler_path: str,
        threshold_path: str,
        config: Dict[str, Any]
    ):
        """
        Initialize SIU Validator

        Args:
            yolo_model_path: Path to YOLO model weights
            siu_model_path: Path to trained SIU classifier
            scaler_path: Path to fitted feature scaler
            threshold_path: Path to instance score threshold
            config: Configuration dictionary
        """
        self.config = config
        self.class_names = config.get('classes', [])

        # Load YOLO model
        logger.info(f"Loading YOLO model from {yolo_model_path}")
        if YOLO is None:
            raise ImportError("ultralytics package is required for YOLO inference")
        self.yolo_model = YOLO(yolo_model_path)

        # Load SIU model
        logger.info(f"Loading SIU model from {siu_model_path}")
        self.siu_model = joblib.load(siu_model_path)

        # Load scaler
        logger.info(f"Loading feature scaler from {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # Load threshold
        logger.info(f"Loading threshold from {threshold_path}")
        with open(threshold_path, 'r') as f:
            self.threshold = float(f.read().strip())

        logger.info(f"SIU Validator initialized (threshold={self.threshold:.2f})")

    def run_yolo_detection(
        self,
        image_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Run YOLO object detection on image

        Args:
            image_path: Path to input image
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS

        Returns:
            List of detection dictionaries
        """
        yolo_config = self.config.get('yolo', {})
        conf = confidence_threshold or yolo_config.get('confidence_threshold', 0.25)
        iou = iou_threshold or yolo_config.get('iou_threshold', 0.45)
        device = yolo_config.get('device', '')

        # Run detection
        results = self.yolo_model.predict(
            image_path,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False
        )

        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    # Convert to normalized YOLO format
                    img_height, img_width = result.orig_shape
                    x_center = ((x1 + x2) / 2) / img_width
                    y_center = ((y1 + y2) / 2) / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height

                    detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}",
                        'confidence': confidence,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2),
                    })

        logger.info(f"YOLO detected {len(detections)} objects")
        return detections

    def validate_structure(
        self,
        detections: List[Dict[str, Any]],
        img_width: int,
        img_height: int
    ) -> Dict[str, Any]:
        """
        Validate structural correctness using SIU model

        Args:
            detections: List of YOLO detections
            img_width: Image width
            img_height: Image height

        Returns:
            Validation results dictionary
        """
        if not detections:
            logger.warning("No detections to validate")
            return {
                'instance_score': 0.0,
                'is_correct_structure': False,
                'num_detections': 0,
                'validated': False
            }

        # Extract geometric features
        features = extract_geometric_features(
            detections,
            img_width,
            img_height,
            self.config
        )

        if len(features) == 0:
            logger.warning("Failed to extract features")
            return {
                'instance_score': 0.0,
                'is_correct_structure': False,
                'num_detections': len(detections),
                'validated': False
            }

        # Pad/truncate to match training feature length
        # This is a simplified approach; in production, you'd save the expected length
        features_padded = features

        # Scale features
        features_scaled = self.scaler.transform(features_padded.reshape(1, -1))

        # Predict
        prediction = self.siu_model.predict(features_scaled)[0]
        prediction_proba = self.siu_model.predict_proba(features_scaled)[0]

        # Instance score (probability of correct structure)
        instance_score = float(prediction_proba[1])

        # Determine if structure is correct based on threshold
        is_correct = instance_score >= self.threshold

        validation_result = {
            'instance_score': instance_score,
            'is_correct_structure': is_correct,
            'num_detections': len(detections),
            'validated': True,
            'prediction_proba': prediction_proba.tolist()
        }

        logger.info(f"Structure validation: score={instance_score:.3f}, "
                   f"correct={is_correct} (threshold={self.threshold:.2f})")

        return validation_result


def run_inference_pipeline(
    image_path: str,
    yolo_model: YOLO,
    siu_model: Any,
    scaler: Any,
    instance_score_threshold: float,
    config: Dict[str, Any],
    save_visualization: bool = True,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Run complete inference pipeline

    Args:
        image_path: Path to input image
        yolo_model: Loaded YOLO model
        siu_model: Loaded SIU classifier
        scaler: Loaded feature scaler
        instance_score_threshold: Threshold for structure validation
        config: Configuration dictionary
        save_visualization: Whether to save visualization
        output_dir: Output directory for results

    Returns:
        Dictionary containing complete inference results
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img_height, img_width = img.shape[:2]

    # Create validator
    validator = SIUValidator(
        yolo_model_path=yolo_model,
        siu_model_path=siu_model,
        scaler_path=scaler,
        threshold_path=instance_score_threshold,
        config=config
    )

    # Run YOLO detection
    detections = validator.run_yolo_detection(image_path)

    # Validate structure
    validation_result = validator.validate_structure(
        detections,
        img_width,
        img_height
    )

    # Prepare results
    results = {
        'image_path': image_path,
        'image_size': (img_width, img_height),
        'yolo_predictions': detections,
        'siu_validation': validation_result
    }

    # Visualize if requested
    if save_visualization and detections:
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Draw detections
        img_vis = visualize_boxes(
            img,
            detections,
            config.get('classes', []),
            title=f"Instance Score: {validation_result['instance_score']:.3f}",
            show_confidence=True
        )

        # Add validation status
        status_text = "✓ CORRECT STRUCTURE" if validation_result['is_correct_structure'] else "✗ INCORRECT STRUCTURE"
        color = (0, 255, 0) if validation_result['is_correct_structure'] else (0, 0, 255)

        cv2.putText(
            img_vis,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2
        )

        score_text = f"Score: {validation_result['instance_score']:.3f} (Threshold: {instance_score_threshold:.2f})"
        cv2.putText(
            img_vis,
            score_text,
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Save
        img_name = Path(image_path).stem
        output_path = os.path.join(vis_dir, f"{img_name}_validated.jpg")
        cv2.imwrite(output_path, img_vis)
        logger.info(f"Visualization saved to {output_path}")

        results['visualization_path'] = output_path

    return results


def main_inference(
    image_path: str,
    config_path: str = "config/config.yaml",
    model_version: str = "latest"
) -> Dict[str, Any]:
    """
    Main inference function

    Args:
        image_path: Path to input image
        config_path: Path to configuration file
        model_version: Model version to use ('latest' or timestamp)

    Returns:
        Inference results dictionary
    """
    # Load configuration
    config = load_config(config_path)
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("SIU INFERENCE PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input image: {image_path}")

    # Model paths
    models_dir = config.get('output', {}).get('models_dir', 'models')

    if model_version == "latest":
        yolo_model_path = config.get('yolo', {}).get('model_name', 'yolov8n.pt')
        siu_model_path = os.path.join(models_dir, 'siu_model_latest.pkl')
        scaler_path = os.path.join(models_dir, 'scaler_latest.pkl')
        threshold_path = os.path.join(models_dir, 'threshold_latest.txt')
    else:
        yolo_model_path = config.get('yolo', {}).get('model_name', 'yolov8n.pt')
        siu_model_path = os.path.join(models_dir, f'siu_model_{model_version}.pkl')
        scaler_path = os.path.join(models_dir, f'scaler_{model_version}.pkl')
        threshold_path = os.path.join(models_dir, f'threshold_{model_version}.txt')

    # Check if SIU model exists
    if not os.path.exists(siu_model_path):
        raise FileNotFoundError(
            f"SIU model not found: {siu_model_path}\n"
            "Please train the model first using: python main.py train"
        )

    # Load threshold
    with open(threshold_path, 'r') as f:
        threshold = float(f.read().strip())

    # Run inference
    output_dir = config.get('output', {}).get('results_dir', 'outputs')

    results = run_inference_pipeline(
        image_path,
        yolo_model_path,
        siu_model_path,
        scaler_path,
        threshold,
        config,
        save_visualization=True,
        output_dir=output_dir
    )

    # Log summary
    logger.info("=" * 60)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Detected objects: {len(results['yolo_predictions'])}")
    logger.info(f"Instance score: {results['siu_validation']['instance_score']:.3f}")
    logger.info(f"Structure valid: {results['siu_validation']['is_correct_structure']}")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    results = main_inference(image_path)
