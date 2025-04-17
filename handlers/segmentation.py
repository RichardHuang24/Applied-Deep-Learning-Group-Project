import logging
from pathlib import Path
from train import train_segmentation # Assuming train.py is in the parent directory or PYTHONPATH is set
import time
from utils.logging import setup_logging

def handle_train_segmentation(args, mask_dir=None):
    """
    Handles the command line arguments and initiates the segmentation model training.

    Args:
        args: Parsed arguments from argparse, expected to have:
            - config (dict): Loaded configuration.
            - supervision (str): Supervision type (e.g., 'weak_gradcam').
            - experiment_name (str): Name for the experiment run.
            - backbone (str): Backbone for the segmentation model.
            - config_path (str): Path to the config file (used by train_segmentation).
    """
    config = args.config
    experiment_name = args.experiment_name
    supervision = args.supervision

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"{args.backbone}_{args.init}_{args.cam}_{timestamp}"
    
    mask_dir = mask_dir if mask_dir is None else mask_dir

    # Define output directory based on experiment name
    output_dir = Path(config['paths']['outputs']) / "experiments" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logger = logging.getLogger("Train Segmentation")
    logger.info(f"Starting segmentation training for experiment: {experiment_name}")
    logger.info(f"Supervision type: {supervision}")
    logger.info(f"Using pseudo masks from: {mask_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Call the main training function from train.py
    try:
        segmentation_model_path = train_segmentation(
            config=config, # Pass the config path itself
            supervision=supervision,
            pseudo_masks_dir=mask_dir,
            experiment_name=experiment_name,
            output_dir=output_dir  # Pass the specific output directory
        )
        logger.info(f"Segmentation model training complete. Model saved to: {segmentation_model_path}")
        return segmentation_model_path # Return path for potential chaining (e.g., in run_all)
    except Exception as e:
        logger.error(f"Error during segmentation training for experiment {experiment_name}: {e}", exc_info=True)
        raise # Re-raise the exception after logging