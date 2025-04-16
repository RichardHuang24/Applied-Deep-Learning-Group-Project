from pathlib import Path
import logging
from utils.load_config import load_config
from utils.logging import setup_logging
from generate_masks import generate_masks
import time

def handle_generate_masks(args, model_path=None):
    config = args.config

    if args.experiment_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.experiment_name = args.experiment_name or f"{args.backbone}_{args.init}_{args.cam}_{timestamp}"
    
    output_dir = Path(config['paths']['outputs']) / "experiments" / args.experiment_name / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging (once per experiment ideally)
    setup_logging(output_dir)
    logger = logging.getLogger("Generate Masks")

    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Backbone: {args.backbone}, Init: {args.init}, CAM: {args.cam}")

    model_path = Path(model_path) if model_path else Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found at: {model_path}")
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    threshold = config.get("models", {}).get("cam", {}).get("threshold", 0.4)
    logger.info(f"Using CAM threshold: {threshold}")

    masks_dir = generate_masks(
        config=config,
        method=args.cam,
        classifier_path=model_path,
        output_dir=output_dir,
        threshold=threshold,
        args=args
    )

    logger.info(f"Masks saved to: {masks_dir}")

    return masks_dir
