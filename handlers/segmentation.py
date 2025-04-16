import logging
from pathlib import Path
from train import train_segmentation # Assuming train.py is in the parent directory or PYTHONPATH is set

logger = logging.getLogger(__name__)

def handle_train_segmentation(args):
    """
    Handles the command line arguments and initiates the segmentation model training.

    Args:
        args: Parsed arguments from argparse, expected to have:
            - config (dict): Loaded configuration.
            - supervision (str): Supervision type (e.g., 'weak_gradcam').
            - pseudo_masks_dir (str): Path to the directory with pseudo masks.
            - experiment_name (str): Name for the experiment run.
            - backbone (str): Backbone for the segmentation model.
            - config_path (str): Path to the config file (used by train_segmentation).
    """
    config = args.config
    experiment_name = args.experiment_name
    supervision = args.supervision

    # Define output directory based on experiment name
    output_dir = Path(config['paths']['outputs']) / "segmentation" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting segmentation training for experiment: {experiment_name}")
    logger.info(f"Supervision type: {supervision}")
    logger.info(f"Using pseudo masks from: {args.pseudo_masks_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Call the main training function from train.py
    try:
        segmentation_model_path = train_segmentation(
            config_path=args.config_path, # Pass the config path itself
            supervision=supervision,
            pseudo_masks_dir=args.pseudo_masks_dir,
            output_dir=output_dir, # Pass the specific output directory
            # num_epochs can be read from config inside train_segmentation
        )
        logger.info(f"Segmentation model training complete. Model saved to: {segmentation_model_path}")
        return segmentation_model_path # Return path for potential chaining (e.g., in run_all)
    except Exception as e:
        logger.error(f"Error during segmentation training for experiment {experiment_name}: {e}", exc_info=True)
        raise # Re-raise the exception after logging