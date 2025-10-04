"""
Utility functions for SIU Object Detection Validator

Includes:
- YOLO annotation parsing
- Configuration loading
- Logging setup
- File I/O helpers
- Visualization utilities
"""

import os
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from datetime import datetime


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        config: Configuration dictionary with logging settings

    Returns:
        Configured logger instance
    """
    log_config = config.get('logging', {})
    log_level = getattr(logging, log_config.get('level', 'INFO'))
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'logs/siu.log')

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger('SIU')
    return logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_yolo_annotation(label_path: str, img_width: int, img_height: int) -> List[Dict[str, Any]]:
    """
    Load YOLO format annotation file

    YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)

    Args:
        label_path: Path to YOLO label file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of bounding box dictionaries with keys:
        - class_id: int
        - x_center: float (normalized)
        - y_center: float (normalized)
        - width: float (normalized)
        - height: float (normalized)
        - x1, y1, x2, y2: int (absolute pixel coordinates)
    """
    if not os.path.exists(label_path):
        return []

    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # Convert to absolute coordinates
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)

            boxes.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
            })

    return boxes


def save_yolo_annotation(boxes: List[Dict[str, Any]], label_path: str) -> None:
    """
    Save bounding boxes in YOLO format

    Args:
        boxes: List of bounding box dictionaries
        label_path: Path to save YOLO label file
    """
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, 'w') as f:
        for box in boxes:
            line = f"{box['class_id']} {box['x_center']:.6f} {box['y_center']:.6f} "
            line += f"{box['width']:.6f} {box['height']:.6f}\n"
            f.write(line)


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get image dimensions without loading full image

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (width, height)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    height, width = img.shape[:2]
    return width, height


def load_image_with_boxes(
    image_path: str,
    label_path: str
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load image and its corresponding YOLO annotations

    Args:
        image_path: Path to image file
        label_path: Path to YOLO label file

    Returns:
        Tuple of (image array, list of bounding boxes)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    height, width = img.shape[:2]
    boxes = load_yolo_annotation(label_path, width, height)

    return img, boxes


def visualize_boxes(
    image: np.ndarray,
    boxes: List[Dict[str, Any]],
    class_names: List[str],
    title: str = "Detection Results",
    show_confidence: bool = False,
    thickness: int = 2,
    font_scale: float = 0.5,
    colors: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Visualize bounding boxes on image

    Args:
        image: Image array (BGR format)
        boxes: List of bounding box dictionaries
        class_names: List of class names
        title: Title for visualization
        show_confidence: Whether to show confidence scores
        thickness: Box line thickness
        font_scale: Font scale for labels
        colors: Optional dict mapping class_id to BGR color tuple

    Returns:
        Image with drawn bounding boxes
    """
    img_vis = image.copy()

    # Generate random colors if not provided
    if colors is None:
        np.random.seed(42)
        colors = {i: tuple(map(int, np.random.randint(0, 255, 3)))
                 for i in range(len(class_names))}

    for box in boxes:
        class_id = box['class_id']
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

        # Get color for this class
        color = colors.get(class_id, (0, 255, 0))

        # Draw bounding box
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)

        # Prepare label
        label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        if show_confidence and 'confidence' in box:
            label += f" {box['confidence']:.2f}"

        # Draw label background
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(
            img_vis,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            img_vis,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )

    return img_vis


def calculate_iou(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes

    Args:
        box1: First bounding box dictionary
        box2: Second bounding box dictionary

    Returns:
        IoU value (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
    x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']

    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def get_dataset_files(
    images_dir: str,
    labels_dir: str,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')
) -> List[Tuple[str, str]]:
    """
    Get matched pairs of image and label files

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
        extensions: Tuple of valid image extensions

    Returns:
        List of (image_path, label_path) tuples
    """
    dataset = []

    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(extensions):
            continue

        img_path = os.path.join(images_dir, img_name)

        # Get corresponding label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        if os.path.exists(label_path):
            dataset.append((img_path, label_path))

    return dataset


def create_output_dirs(config: Dict[str, Any]) -> None:
    """
    Create output directories based on configuration

    Args:
        config: Configuration dictionary
    """
    output_config = config.get('output', {})

    dirs = [
        output_config.get('models_dir', 'models'),
        output_config.get('results_dir', 'outputs'),
        output_config.get('visualizations_dir', 'outputs/visualizations'),
        output_config.get('logs_dir', 'logs'),
    ]

    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def get_timestamp() -> str:
    """
    Get current timestamp as string

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
