#!/usr/bin/env python3
"""
Main entry point for SIU Object Detection Validator

This script provides a command-line interface for:
- Training the SIU model
- Running inference on images
- Batch processing multiple images
- Evaluating model performance

Usage:
    python main.py train [--config CONFIG_PATH]
    python main.py inference IMAGE_PATH [--config CONFIG_PATH] [--model-version VERSION]
    python main.py batch INPUT_DIR [--config CONFIG_PATH] [--output OUTPUT_DIR]
    python main.py evaluate [--config CONFIG_PATH]
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import main_train
from src.inference import main_inference
from src.utils import load_config, setup_logging

logger = logging.getLogger('SIU.Main')


def train_command(args):
    """Execute training command"""
    print("=" * 70)
    print("  SIU OBJECT DETECTION VALIDATOR - TRAINING MODE")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print()

    try:
        main_train(args.config)
        print("\n✓ Training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        logger.exception("Training failed")
        return 1


def inference_command(args):
    """Execute inference command"""
    print("=" * 70)
    print("  SIU OBJECT DETECTION VALIDATOR - INFERENCE MODE")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print(f"Input image: {args.image}")
    print(f"Model version: {args.model_version}")
    print()

    if not os.path.exists(args.image):
        print(f"✗ Error: Image not found: {args.image}")
        return 1

    try:
        results = main_inference(
            args.image,
            config_path=args.config,
            model_version=args.model_version
        )

        print("\n" + "=" * 70)
        print("  INFERENCE RESULTS")
        print("=" * 70)
        print(f"Detected objects: {len(results['yolo_predictions'])}")
        print(f"Instance score: {results['siu_validation']['instance_score']:.4f}")
        print(f"Structure valid: {'✓ YES' if results['siu_validation']['is_correct_structure'] else '✗ NO'}")

        if 'visualization_path' in results:
            print(f"Visualization saved: {results['visualization_path']}")

        print()

        # Show detected objects
        if results['yolo_predictions']:
            print("Detected objects:")
            for i, det in enumerate(results['yolo_predictions'], 1):
                print(f"  {i}. {det['class_name']}: confidence={det['confidence']:.3f}")

        print("\n✓ Inference completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Inference failed: {e}")
        logger.exception("Inference failed")
        return 1


def batch_command(args):
    """Execute batch inference command"""
    print("=" * 70)
    print("  SIU OBJECT DETECTION VALIDATOR - BATCH MODE")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output}")
    print()

    if not os.path.isdir(args.input_dir):
        print(f"✗ Error: Input directory not found: {args.input_dir}")
        return 1

    # Find all images in input directory
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(args.input_dir).glob(f"*{ext}"))
        image_files.extend(Path(args.input_dir).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"✗ Error: No images found in {args.input_dir}")
        return 1

    print(f"Found {len(image_files)} images to process\n")

    # Process each image
    success_count = 0
    fail_count = 0
    results_summary = []

    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {image_path.name}...", end=" ")

        try:
            results = main_inference(
                str(image_path),
                config_path=args.config,
                model_version=args.model_version
            )

            success_count += 1
            results_summary.append({
                'image': image_path.name,
                'num_objects': len(results['yolo_predictions']),
                'instance_score': results['siu_validation']['instance_score'],
                'is_correct': results['siu_validation']['is_correct_structure']
            })
            print("✓")

        except Exception as e:
            fail_count += 1
            print(f"✗ ({e})")
            logger.error(f"Failed to process {image_path}: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("  BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

    if results_summary:
        print("\nResults:")
        correct_count = sum(1 for r in results_summary if r['is_correct'])
        incorrect_count = len(results_summary) - correct_count

        print(f"  Correct structure: {correct_count}")
        print(f"  Incorrect structure: {incorrect_count}")
        print(f"  Average instance score: {sum(r['instance_score'] for r in results_summary) / len(results_summary):.4f}")

    print()
    return 0 if fail_count == 0 else 1


def evaluate_command(args):
    """Execute evaluation command"""
    print("=" * 70)
    print("  SIU OBJECT DETECTION VALIDATOR - EVALUATION MODE")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print()

    print("Evaluation on test set not yet implemented.")
    print("Please use the training script which includes evaluation on the test split.")

    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SIU Object Detection Validator - Structured Instance Understanding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the SIU model
  python main.py train

  # Run inference on a single image
  python main.py inference path/to/image.jpg

  # Batch process multiple images
  python main.py batch path/to/images/ --output results/

  # Use custom configuration
  python main.py train --config my_config.yaml
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the SIU model')
    train_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    train_parser.set_defaults(func=train_command)

    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference on an image')
    inference_parser.add_argument(
        'image',
        type=str,
        help='Path to input image'
    )
    inference_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    inference_parser.add_argument(
        '--model-version',
        type=str,
        default='latest',
        help='Model version to use (default: latest)'
    )
    inference_parser.set_defaults(func=inference_command)

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple images')
    batch_parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing input images'
    )
    batch_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    batch_parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    batch_parser.add_argument(
        '--model-version',
        type=str,
        default='latest',
        help='Model version to use (default: latest)'
    )
    batch_parser.set_defaults(func=batch_command)

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test set')
    eval_parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    eval_parser.set_defaults(func=evaluate_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
