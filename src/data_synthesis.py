"""
Data Synthesis Module for SIU Model

Implements the four methods for synthesizing erroneous (incorrect structure) data
as described in Section 3.1.1 of the SIU paper:

1. Class Transform: Swap class labels between objects
2. Random Add: Add spurious/duplicate bounding boxes
3. Random Delete: Remove bounding boxes
4. Box Shift: Shift bounding boxes to incorrect positions

These methods create training data for the binary classifier to distinguish
between correct and incorrect object structures.
"""

import numpy as np
import copy
from typing import List, Dict, Any
import logging

logger = logging.getLogger('SIU.DataSynthesis')


def synthesize_erroneous_data(
    annotations: List[Dict[str, Any]],
    method: str,
    config: Dict[str, Any],
    img_width: int = 640,
    img_height: int = 640,
    num_classes: int = 20
) -> List[Dict[str, Any]]:
    """
    Synthesize erroneous (incorrect structure) data using specified method

    Args:
        annotations: List of correct bounding box dictionaries
        method: Synthesis method name
                - 'class_transform': Swap class labels
                - 'random_add': Add spurious boxes
                - 'random_delete': Remove boxes
                - 'box_shift': Shift boxes to wrong positions
        config: Configuration dictionary with synthesis parameters
        img_width: Image width for coordinate calculations
        img_height: Image height for coordinate calculations
        num_classes: Number of object classes

    Returns:
        List of erroneous bounding box dictionaries
    """
    if not annotations:
        logger.warning("Empty annotations provided for synthesis")
        return annotations

    synthesis_config = config.get('synthesis', {})

    if method == 'class_transform':
        return class_transform(annotations, synthesis_config, num_classes)
    elif method == 'random_add':
        return random_add(annotations, synthesis_config, img_width, img_height, num_classes)
    elif method == 'random_delete':
        return random_delete(annotations, synthesis_config)
    elif method == 'box_shift':
        return box_shift(annotations, synthesis_config, img_width, img_height)
    else:
        raise ValueError(f"Unknown synthesis method: {method}")


def class_transform(
    annotations: List[Dict[str, Any]],
    config: Dict[str, Any],
    num_classes: int
) -> List[Dict[str, Any]]:
    """
    Class Transform: Swap class labels between objects

    This creates structurally incorrect data by assigning wrong class labels
    to objects while keeping their spatial locations intact.

    Reference: Section 3.1.1, Method 1 of the paper

    Args:
        annotations: Original bounding boxes
        config: Configuration with 'class_transform' parameters
        num_classes: Number of object classes

    Returns:
        Annotations with transformed class labels
    """
    erroneous = copy.deepcopy(annotations)
    transform_config = config.get('class_transform', {})
    probability = transform_config.get('probability', 0.3)

    for box in erroneous:
        if np.random.random() < probability:
            # Swap to a different random class
            original_class = box['class_id']
            new_class = np.random.randint(0, num_classes)

            # Ensure it's actually different
            while new_class == original_class and num_classes > 1:
                new_class = np.random.randint(0, num_classes)

            box['class_id'] = new_class

    logger.debug(f"Class transform: Modified {len(erroneous)} boxes")
    return erroneous


def random_add(
    annotations: List[Dict[str, Any]],
    config: Dict[str, Any],
    img_width: int,
    img_height: int,
    num_classes: int
) -> List[Dict[str, Any]]:
    """
    Random Add: Add spurious/duplicate bounding boxes

    This creates structurally incorrect data by adding boxes that shouldn't exist,
    such as duplicates or boxes in random locations.

    Reference: Section 3.1.1, Method 2 of the paper

    Args:
        annotations: Original bounding boxes
        config: Configuration with 'random_add' parameters
        img_width: Image width
        img_height: Image height
        num_classes: Number of object classes

    Returns:
        Annotations with additional spurious boxes
    """
    erroneous = copy.deepcopy(annotations)
    add_config = config.get('random_add', {})

    min_boxes = add_config.get('min_boxes', 1)
    max_boxes = add_config.get('max_boxes', 3)
    overlap_threshold = add_config.get('overlap_threshold', 0.3)

    num_to_add = np.random.randint(min_boxes, max_boxes + 1)

    for _ in range(num_to_add):
        # Two strategies: duplicate existing box or create random box
        if len(annotations) > 0 and np.random.random() < 0.5:
            # Strategy 1: Duplicate and slightly shift an existing box
            source_box = copy.deepcopy(np.random.choice(annotations))

            # Shift the box slightly
            shift_x = np.random.uniform(-0.1, 0.1)
            shift_y = np.random.uniform(-0.1, 0.1)

            source_box['x_center'] = np.clip(source_box['x_center'] + shift_x, 0, 1)
            source_box['y_center'] = np.clip(source_box['y_center'] + shift_y, 0, 1)

            # Update absolute coordinates
            x1 = int((source_box['x_center'] - source_box['width'] / 2) * img_width)
            y1 = int((source_box['y_center'] - source_box['height'] / 2) * img_height)
            x2 = int((source_box['x_center'] + source_box['width'] / 2) * img_width)
            y2 = int((source_box['y_center'] + source_box['height'] / 2) * img_height)

            source_box.update({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

            erroneous.append(source_box)

        else:
            # Strategy 2: Create a completely random box
            width = np.random.uniform(0.05, 0.3)
            height = np.random.uniform(0.05, 0.3)
            x_center = np.random.uniform(width / 2, 1 - width / 2)
            y_center = np.random.uniform(height / 2, 1 - height / 2)

            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)

            new_box = {
                'class_id': np.random.randint(0, num_classes),
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
            }

            erroneous.append(new_box)

    logger.debug(f"Random add: Added {num_to_add} boxes, total now {len(erroneous)}")
    return erroneous


def random_delete(
    annotations: List[Dict[str, Any]],
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Random Delete: Remove bounding boxes

    This creates structurally incorrect data by removing objects that should exist,
    creating incomplete detections.

    Reference: Section 3.1.1, Method 3 of the paper

    Args:
        annotations: Original bounding boxes
        config: Configuration with 'random_delete' parameters

    Returns:
        Annotations with some boxes removed
    """
    if len(annotations) <= 1:
        logger.warning("Cannot delete from annotations with <= 1 box")
        return copy.deepcopy(annotations)

    erroneous = copy.deepcopy(annotations)
    delete_config = config.get('random_delete', {})

    min_delete = delete_config.get('min_delete', 1)
    max_delete = delete_config.get('max_delete', 3)

    # Don't delete all boxes
    num_to_delete = min(
        np.random.randint(min_delete, max_delete + 1),
        len(erroneous) - 1
    )

    # Randomly select boxes to delete
    indices_to_delete = np.random.choice(
        len(erroneous),
        size=num_to_delete,
        replace=False
    )

    # Remove selected boxes (in reverse order to maintain indices)
    for idx in sorted(indices_to_delete, reverse=True):
        erroneous.pop(idx)

    logger.debug(f"Random delete: Removed {num_to_delete} boxes, {len(erroneous)} remaining")
    return erroneous


def box_shift(
    annotations: List[Dict[str, Any]],
    config: Dict[str, Any],
    img_width: int,
    img_height: int
) -> List[Dict[str, Any]]:
    """
    Box Shift: Shift bounding boxes to incorrect positions

    This creates structurally incorrect data by moving boxes to wrong spatial
    locations while keeping their class labels correct.

    Reference: Section 3.1.1, Method 4 of the paper

    Args:
        annotations: Original bounding boxes
        config: Configuration with 'box_shift' parameters
        img_width: Image width
        img_height: Image height

    Returns:
        Annotations with shifted box positions
    """
    erroneous = copy.deepcopy(annotations)
    shift_config = config.get('box_shift', {})

    shift_range = shift_config.get('shift_range', [0.2, 0.5])
    probability = shift_config.get('probability', 0.3)

    for box in erroneous:
        if np.random.random() < probability:
            # Random shift amount (as fraction of image size)
            shift_amount = np.random.uniform(shift_range[0], shift_range[1])

            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            shift_x = shift_amount * np.cos(angle)
            shift_y = shift_amount * np.sin(angle)

            # Apply shift
            new_x_center = box['x_center'] + shift_x
            new_y_center = box['y_center'] + shift_y

            # Keep box within image bounds
            half_width = box['width'] / 2
            half_height = box['height'] / 2

            new_x_center = np.clip(new_x_center, half_width, 1 - half_width)
            new_y_center = np.clip(new_y_center, half_height, 1 - half_height)

            # Update box coordinates
            box['x_center'] = new_x_center
            box['y_center'] = new_y_center

            # Update absolute coordinates
            x1 = int((new_x_center - half_width) * img_width)
            y1 = int((new_y_center - half_height) * img_height)
            x2 = int((new_x_center + half_width) * img_width)
            y2 = int((new_y_center + half_height) * img_height)

            box.update({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    logger.debug(f"Box shift: Shifted positions for {len(erroneous)} boxes")
    return erroneous


def generate_synthetic_dataset(
    annotations_list: List[List[Dict[str, Any]]],
    config: Dict[str, Any],
    img_width: int = 640,
    img_height: int = 640,
    num_classes: int = 20
) -> tuple:
    """
    Generate complete synthetic dataset with correct and incorrect structures

    Args:
        annotations_list: List of annotation sets (each is a list of boxes for one image)
        config: Configuration dictionary
        img_width: Image width
        img_height: Image height
        num_classes: Number of classes

    Returns:
        Tuple of (correct_samples, incorrect_samples)
        Each is a list of annotation sets
    """
    synthesis_config = config.get('synthesis', {})
    synthesis_ratio = synthesis_config.get('synthesis_ratio', 1.0)

    correct_samples = annotations_list
    incorrect_samples = []

    methods = ['class_transform', 'random_add', 'random_delete', 'box_shift']

    for annotations in annotations_list:
        if not annotations:
            continue

        # Generate incorrect samples based on ratio
        num_incorrect = int(synthesis_ratio * len(methods))

        # Use all methods or sample them
        selected_methods = np.random.choice(
            methods,
            size=min(num_incorrect, len(methods)),
            replace=False
        )

        for method in selected_methods:
            try:
                erroneous = synthesize_erroneous_data(
                    annotations,
                    method,
                    config,
                    img_width,
                    img_height,
                    num_classes
                )
                incorrect_samples.append(erroneous)
            except Exception as e:
                logger.error(f"Error in {method} synthesis: {e}")
                continue

    logger.info(f"Generated {len(correct_samples)} correct and "
                f"{len(incorrect_samples)} incorrect samples")

    return correct_samples, incorrect_samples
