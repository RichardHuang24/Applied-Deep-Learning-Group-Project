# GenAI is used for rephrasing comments and debugging.
from train import train_classifier
from utils.load_config import load_config
from utils.logging import setup_logging

from pathlib import Path
import time
import os

def handle_train_classifier(args):
    config = args.config

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"{args.backbone}_{args.init}_{args.cam}_{timestamp}"
    output_dir = Path(config['paths']['outputs']) / "experiments" / experiment_name

    setup_logging(output_dir)

    if (Path(config['paths']['outputs']) / "classifier.pth").exists():
        print(f"Classifier already exists at {Path(config['paths']['outputs']) / 'classifier.pth'}. Skipping training.")
        return os.path.join(Path(config['paths']['outputs']), 'classifier.pth'), experiment_name
    if args.cam == "ccam":
        print("CCAM does not need classifier training. Skipping.")
        return None, experiment_name

    model_path = train_classifier(
        config=config,
        experiment=experiment_name,
        output_dir=output_dir
    )

    print(f"Classifier saved to {model_path}")

    return model_path, experiment_name