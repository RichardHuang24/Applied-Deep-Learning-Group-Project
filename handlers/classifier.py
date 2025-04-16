from train import train_classifier
from utils.load_config import load_config
from utils.logging import setup_logging

from pathlib import Path
import time

def handle_train_classifier(args):
    config = load_config(args.config_path)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"{args.backbone}_{args.init}_{args.cam}_{timestamp}"
    output_dir = Path(config['paths']['outputs']) / "experiments" / experiment_name

    setup_logging(output_dir)

    model_path = train_classifier(
        config=config,
        experiment=experiment_name,
        output_dir=output_dir
    )

    print(f"Classifier saved to {model_path}")

    return model_path, experiment_name