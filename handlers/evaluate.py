import json
import logging
from pathlib import Path
from evaluate import evaluate
import time

logger = logging.getLogger(__name__)

def handle_evaluate(args):

    checkpoint_path = Path(args.checkpoint)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"{args.backbone}_{args.init}_{args.cam}_{timestamp}"
    
    # Define output directory based on experiment name
    output_dir = Path(args.config['paths']['outputs']) / "experiments" / experiment_name / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate(
        config=args.config,
        args=args,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )

    print("Evaluation Metrics:")
    print(json.dumps(metrics, indent=4))
