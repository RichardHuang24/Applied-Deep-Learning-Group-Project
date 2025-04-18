import argparse
from handlers.classifier import handle_train_classifier
from handlers.masks import handle_generate_masks
from handlers.segmentation import handle_train_segmentation
from handlers.evaluate import handle_evaluate
import json
from pathlib import Path
import os
import random
import numpy as np
import torch
from utils.download import download_dataset

def main():
    # Common parser for all subcommands
    common_parser = argparse.ArgumentParser(add_help=False)

    # Main CLI parser
    parser = argparse.ArgumentParser(description="WSSS CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Download-only Subcommand
    parser_download = subparsers.add_parser("download", parents=[common_parser], help="Download dataset and exit")
    parser_download.add_argument("--config_path", help="Path to the config file", default="config.json")
    parser_download.set_defaults(func=handle_download)

    # --- Train Classifier
    parser_classifier = subparsers.add_parser("train_classifier", parents=[common_parser], help="Train image classifier")
    parser_classifier.add_argument("--config_path", help="Path to the config file", default="config.json")
    parser_classifier.add_argument("--backbone", help="Backbone model for the classifier", default="resnet50")
    parser_classifier.add_argument("--init", help="Initialization method for the classifier", default="imagenet")
    parser_classifier.add_argument("--cam", choices=["gradcam", "cam", "gradcam+ccam", "cam+ccam", "ccam"], help="CAM method to use", default="gradcam")
    parser_classifier.add_argument("--experiment_name", default=None)
    parser_classifier.set_defaults(func=handle_train_classifier)

    # --- Generate Masks
    parser_masks = subparsers.add_parser("generate_masks", parents=[common_parser], help="Generate CAM-based pseudo masks")
    parser_masks.add_argument("--config_path", help="Path to the config file", default="config.json")
    parser_masks.add_argument("--cam", choices=["gradcam", "cam", "gradcam+ccam", "cam+ccam", "ccam"], help="CAM method to use", default="gradcam")
    parser_masks.add_argument("--model_path", required=True, help="Path to the trained model checkpoint")
    parser_masks.add_argument("--backbone", help="Backbone model for the classifier", default="resnet50")
    parser_masks.add_argument("--experiment_name", default=None)
    parser_masks.add_argument("--init", help="Initialization method for the classifier", default="imagenet")
    parser_masks.set_defaults(func=handle_generate_masks)

    # --- Train + Generate
    parser_combo = subparsers.add_parser("train_and_generate", parents=[common_parser], help="Train classifier and generate masks")
    parser_combo.add_argument("--backbone", help="Backbone model for the classifier", default="resnet50")
    parser_combo.add_argument("--init", help="Initialization method for the classifier", default="imagenet")
    parser_combo.add_argument("--config_path", help="Path to the config file", default="config.json")
    parser_combo.add_argument("--cam", choices=["gradcam", "cam", "gradcam+ccam", "cam+ccam", "ccam"], help="CAM method to use", default="gradcam")
    parser_combo.add_argument("--experiment_name", default=None)
    parser_combo.set_defaults(func=handle_train_and_generate_masks)

    # --- Train Segmentation
    parser_segmentation = subparsers.add_parser("train_segmentation", parents=[common_parser], help="Train segmentation model")
    parser_segmentation.add_argument("--config_path", help="Path to the config file", default="config.json")
    parser_segmentation.add_argument("--supervision", choices=["full", "weak_gradcam", "weak_cam"], help="Supervision type", default="weak_cam")
    parser_segmentation.add_argument("--pseudo_masks_dir", default=None, help="Directory containing pseudo masks")
    parser_segmentation.add_argument("--experiment_name", default=None)
    parser_segmentation.add_argument("--init", help="Initialization method for the classifier", default="imagenet")
    parser_segmentation.add_argument("--cam", choices=["gradcam", "cam", "gradcam+ccam", "cam+ccam", "ccam"], help="CAM method to use", default="gradcam")
    parser_segmentation.add_argument("--backbone", help="Backbone model for the classifier", default="resnet50")
    parser_segmentation.set_defaults(func=handle_train_segmentation)

    # --- Run in series
    parser_series = subparsers.add_parser("run_series", parents=[common_parser], help="Run the full WSSS pipeline in series")
    parser_series.add_argument("--backbone", help="Backbone model for the classifier", default="resnet50")
    parser_series.add_argument("--init", help="Initialization method for the classifier", default="imagenet")
    parser_series.add_argument("--cam", choices=["gradcam", "cam", "gradcam+ccam", "cam+ccam", "ccam"], help="CAM method to use", default="gradcam")
    parser_series.add_argument("--experiment_name", default=None)
    parser_series.add_argument("--supervision", choices=["full", "weak_gradcam", "weak_cam"], help="Supervision type", default="weak_gradcam")
    parser_series.add_argument("--config_path", help="Path to the config file", default="config.json")
    parser_series.set_defaults(func=handle_train_series)

    # --- Evaluate
    parser_eval = subparsers.add_parser("evaluate", parents=[common_parser], help="Evaluate segmentation model")
    parser_eval.add_argument("--config_path", help="Path to the config file", default="config.json")
    parser_eval.add_argument("--checkpoint",required=True, help="Path to model checkpoint")
    parser_eval.add_argument("--visualize", help="Save visualization images", default=False, action="store_false")
    parser_eval.add_argument("--experiment_name", default=None)
    parser_eval.add_argument("--cam", choices=["gradcam", "cam", "gradcam+ccam", "cam+ccam", "ccam"], help="CAM method to use", default="gradcam")
    parser_eval.add_argument("--backbone", help="Backbone model for the classifier", default="resnet50")
    parser_eval.add_argument("--init", choices=["random", "imagenet"], help="Initialization method for the classifier", default="imagenet")
    parser_eval.set_defaults(func=handle_evaluate)

    # --- Run All
    parser_all = subparsers.add_parser("run_all", parents=[common_parser], help="Run the full WSSS pipeline")
    parser_all.add_argument("--backbone", help="Backbone model for the classifier", default="resnet50")
    parser_all.add_argument("--init", help="Initialization method for the classifier", default="imagenet")
    parser_all.add_argument("--cam", choices=["gradcam", "cam", "gradcam+ccam", "cam+ccam", "ccam"], help="CAM method to use", default="gradcam")
    parser_all.add_argument("--config_path", help="Path to the config file", default="config.json")
    parser_all.add_argument("--visualize", help="Save visualization images", default=True, action="store_true")
    parser_all.add_argument("--experiment_name", default=None)
    parser_all.add_argument("--supervision", choices=["full", "weak_gradcam", "weak_cam"], help="Supervision type", default="weak_gradcam")
    parser_all.set_defaults(func=handle_run_all)

    # Parse and execute
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        args.config = json.load(f)
    
    args.func(args)
    # Set seed for reproducibility
    set_seed(args.config["training"]["seed"])

def set_seed(seed=42):
    """
    Set seed for reproducibility across Python, NumPy, and PyTorch
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Turn off benchmark for exact reproducibility

    print(f"[Seed Set] {seed}")

def handle_train_series(args):
    """
    Trains the classifier and generates masks using the trained model.
    """
    if args.supervision!='full':
        # Step 1: Train classifier
        model_path, exp_name = handle_train_classifier(args)
        args.experiment_name = exp_name  # Inject for generate step

        # Step 2: Generate masks
        mask_dir = handle_generate_masks(args, model_path)

    # Step 3: train segmentation
    segmentation_model_path = handle_train_segmentation(
        args=args,
        mask_dir=mask_dir
    )

def handle_download(args):
    dataset_dir = Path(args.config["dataset"]["root"])
    print(f"Downloading dataset to {dataset_dir}")
    download_dataset(dataset_dir, seed=42)
    print("Download complete. Exiting.")

def handle_train_and_generate_masks(args):
    """
    Trains the classifier and generates masks using the trained model.
    """
    # Step 1: Train classifier
    model_path, exp_name = handle_train_classifier(args)
    args.experiment_name = exp_name  # Inject for generate step

    # Step 2: Generate masks
    _ = handle_generate_masks(args, model_path)

# Optional: Series runner
def handle_run_all(args):
    """
    Runs the full pipeline: classifier → CAM → masks → segmentation → evaluation
    """
    if args.supervision!='full':
        # Step 1: Train classifier
        model_path, exp_name = handle_train_classifier(args)
        args.experiment_name = exp_name  # Inject for generate step

        # Step 2: Generate masks
        mask_dir = handle_generate_masks(args, model_path)
    mask_dir = None if args.supervision=='full' else mask_dir

    # Step 3: train segmentation
    segmentation_model_path = handle_train_segmentation(
        args=args,
        mask_dir=mask_dir
    )
    
    # Step 4: Evaluate segmentation model
    args.checkpoint = segmentation_model_path
    handle_evaluate(
        args=args
    )
    print(f"Segmentation model saved to {segmentation_model_path}")
    print("All steps completed successfully.")

if __name__ == "__main__":
    main()
