"""
Feature Engineering Module for SIU Model

Implements geometric feature extraction from bounding boxes as described in the paper.
Features include:
- Pairwise distance between box centers
- Pairwise angles between boxes
- Box size ratios
- Relative positions
- IoU values
- Box dimensions

Reference: Section 3.1.2 of the SIU paper
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger('SIU.FeatureEngineering')


def extract_geometric_features(
    bounding_boxes: List[Dict[str, Any]],
    img_width: int = 640,
    img_height: int = 640,
    config: Dict[str, Any] = None
) -> np.ndarray:
    """
    Extract geometric features from bounding boxes for SIU classification

    This function computes pairwise geometric relationships between all detected
    objects, creating a feature vector that captures the structural arrangement.

    Args:
        bounding_boxes: List of bounding box dictionaries with keys:
                       - class_id, x_center, y_center, width, height
                       - x1, y1, x2, y2
        img_width: Image width for normalization
        img_height: Image height for normalization
        config: Feature configuration dictionary

    Returns:
        Feature vector as numpy array
        Shape: (n_features,) where n_features depends on number of boxes and enabled features
    """
    if config is None:
        config = {
            'use_distance': True,
            'use_angle': True,
            'use_size_ratio': True,
            'use_relative_position': True,
            'use_iou': True,
            'use_box_dimensions': True,
        }

    features_config = config.get('features', config)

    n_boxes = len(bounding_boxes)

    if n_boxes == 0:
        logger.warning("No bounding boxes provided for feature extraction")
        return np.array([])

    features = []

    # Image diagonal for distance normalization
    img_diagonal = np.sqrt(img_width**2 + img_height**2)

    # Extract features for each bounding box individually
    for box in bounding_boxes:
        if features_config.get('use_box_dimensions', True):
            # Normalized box dimensions
            features.append(box['width'])
            features.append(box['height'])
            # Box area
            features.append(box['width'] * box['height'])
            # Aspect ratio
            aspect_ratio = box['width'] / (box['height'] + 1e-6)
            features.append(aspect_ratio)

        # Box center position (normalized)
        features.append(box['x_center'])
        features.append(box['y_center'])

    # Extract pairwise features between all boxes
    for i in range(n_boxes):
        for j in range(i + 1, n_boxes):
            box1 = bounding_boxes[i]
            box2 = bounding_boxes[j]

            # Get box centers in pixel coordinates
            cx1 = box1['x_center'] * img_width
            cy1 = box1['y_center'] * img_height
            cx2 = box2['x_center'] * img_width
            cy2 = box2['y_center'] * img_height

            if features_config.get('use_distance', True):
                # Euclidean distance between centers (normalized by image diagonal)
                distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                normalized_distance = distance / img_diagonal
                features.append(normalized_distance)

            if features_config.get('use_angle', True):
                # Angle between box centers (in radians, converted to [-1, 1] range)
                angle = np.arctan2(cy2 - cy1, cx2 - cx1)
                # Normalize to [-1, 1]
                normalized_angle = angle / np.pi
                features.append(normalized_angle)
                # Also add sin and cos for circular continuity
                features.append(np.sin(angle))
                features.append(np.cos(angle))

            if features_config.get('use_relative_position', True):
                # Relative position (dx, dy normalized)
                dx = (cx2 - cx1) / img_width
                dy = (cy2 - cy1) / img_height
                features.append(dx)
                features.append(dy)

            if features_config.get('use_size_ratio', True):
                # Area ratio between boxes
                area1 = box1['width'] * box1['height']
                area2 = box2['width'] * box2['height']
                size_ratio = area1 / (area2 + 1e-6)
                # Log scale to handle large ratios
                features.append(np.log1p(size_ratio))

                # Width and height ratios
                width_ratio = box1['width'] / (box2['width'] + 1e-6)
                height_ratio = box1['height'] / (box2['height'] + 1e-6)
                features.append(np.log1p(width_ratio))
                features.append(np.log1p(height_ratio))

            if features_config.get('use_iou', True):
                # IoU between boxes (spatial overlap)
                iou = calculate_iou_from_boxes(box1, box2)
                features.append(iou)

            # Class relationship features
            class_diff = abs(box1['class_id'] - box2['class_id'])
            features.append(class_diff)
            # Same class indicator
            same_class = 1.0 if box1['class_id'] == box2['class_id'] else 0.0
            features.append(same_class)

    # Add global features
    # Number of objects
    features.append(n_boxes)

    # Average box size
    avg_width = np.mean([box['width'] for box in bounding_boxes])
    avg_height = np.mean([box['height'] for box in bounding_boxes])
    features.append(avg_width)
    features.append(avg_height)

    # Std of box sizes
    std_width = np.std([box['width'] for box in bounding_boxes])
    std_height = np.std([box['height'] for box in bounding_boxes])
    features.append(std_width)
    features.append(std_height)

    # Spatial spread (std of centers)
    std_cx = np.std([box['x_center'] for box in bounding_boxes])
    std_cy = np.std([box['y_center'] for box in bounding_boxes])
    features.append(std_cx)
    features.append(std_cy)

    # Class diversity
    unique_classes = len(set([box['class_id'] for box in bounding_boxes]))
    features.append(unique_classes)

    return np.array(features, dtype=np.float32)


def calculate_iou_from_boxes(box1: Dict[str, Any], box2: Dict[str, Any]) -> float:
    """
    Calculate IoU between two bounding boxes

    Args:
        box1: First bounding box dictionary
        box2: Second bounding box dictionary

    Returns:
        IoU value (0-1)
    """
    x1_1, y1_1, x2_1, y2_1 = box1['x1'], box1['y1'], box1['x2'], box1['y2']
    x1_2, y1_2, x2_2, y2_2 = box2['x1'], box2['y1'], box2['x2'], box2['y2']

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def extract_part_wise_features(
    bounding_boxes: List[Dict[str, Any]],
    img_width: int = 640,
    img_height: int = 640
) -> List[np.ndarray]:
    """
    Extract features for each individual part (for part-level classification)

    This is used during inference to classify each detected part as correct/incorrect

    Args:
        bounding_boxes: List of bounding box dictionaries
        img_width: Image width
        img_height: Image height

    Returns:
        List of feature vectors, one per bounding box
    """
    n_boxes = len(bounding_boxes)
    if n_boxes == 0:
        return []

    part_features = []
    img_diagonal = np.sqrt(img_width**2 + img_height**2)

    for i, target_box in enumerate(bounding_boxes):
        features = []

        # Individual box features
        features.append(target_box['width'])
        features.append(target_box['height'])
        features.append(target_box['width'] * target_box['height'])
        features.append(target_box['width'] / (target_box['height'] + 1e-6))
        features.append(target_box['x_center'])
        features.append(target_box['y_center'])

        # Relationship with all other boxes
        for j, other_box in enumerate(bounding_boxes):
            if i == j:
                continue

            cx1 = target_box['x_center'] * img_width
            cy1 = target_box['y_center'] * img_height
            cx2 = other_box['x_center'] * img_width
            cy2 = other_box['y_center'] * img_height

            # Distance
            distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            features.append(distance / img_diagonal)

            # Angle
            angle = np.arctan2(cy2 - cy1, cx2 - cx1)
            features.append(angle / np.pi)

            # Relative position
            dx = (cx2 - cx1) / img_width
            dy = (cy2 - cy1) / img_height
            features.append(dx)
            features.append(dy)

            # Size ratio
            area1 = target_box['width'] * target_box['height']
            area2 = other_box['width'] * other_box['height']
            features.append(np.log1p(area1 / (area2 + 1e-6)))

            # IoU
            iou = calculate_iou_from_boxes(target_box, other_box)
            features.append(iou)

            # Class relationship
            features.append(abs(target_box['class_id'] - other_box['class_id']))
            features.append(1.0 if target_box['class_id'] == other_box['class_id'] else 0.0)

        # Global context features
        features.append(n_boxes)

        part_features.append(np.array(features, dtype=np.float32))

    return part_features


def pad_or_truncate_features(
    features: np.ndarray,
    target_length: int,
    padding_value: float = 0.0
) -> np.ndarray:
    """
    Pad or truncate feature vector to a fixed length

    This is useful when training classifiers that require fixed-size input

    Args:
        features: Input feature vector
        target_length: Desired length
        padding_value: Value to use for padding

    Returns:
        Feature vector of length target_length
    """
    current_length = len(features)

    if current_length == target_length:
        return features
    elif current_length < target_length:
        # Pad
        padding = np.full(target_length - current_length, padding_value, dtype=features.dtype)
        return np.concatenate([features, padding])
    else:
        # Truncate
        return features[:target_length]


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to zero mean and unit variance

    Args:
        features: Feature vector or matrix

    Returns:
        Normalized features
    """
    if features.ndim == 1:
        mean = np.mean(features)
        std = np.std(features)
        if std > 0:
            return (features - mean) / std
        return features
    else:
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        return (features - mean) / std
