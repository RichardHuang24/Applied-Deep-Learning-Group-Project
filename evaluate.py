import json
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from models.pspnet import create_segmentation_model
from data import data_loaders
from utils.metrics import calculate_metrics
from utils.visualization import visualize_prediction, visualize_results_grid

logger = logging.getLogger(__name__)


def evaluate_model(model, dataloader, device, num_classes=2, visualize=True, output_dir=None):
    model.eval()
    total_pixel_acc, total_miou, total_samples = 0, 0, 0
    all_images, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for images, trimaps, filename in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            trimaps = trimaps.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            metrics = calculate_metrics(preds, trimaps, num_classes)
            total_pixel_acc += metrics['pixel_acc'] * images.size(0)
            total_miou += metrics['miou'] * images.size(0)
            total_samples += images.size(0)

            if visualize and output_dir:
                for i in range(images.size(0)):
                    vis_img = visualize_prediction(images[i].cpu(), preds[i].cpu(), trimaps[i].cpu())
                    vis_img.save(output_dir / f"{filename}_pred.png")

                all_images.extend(images.cpu())
                all_preds.extend(preds.cpu())
                all_targets.extend(trimaps.cpu())

    metrics = {
        'pixel_acc': total_pixel_acc / total_samples,
        'miou': total_miou / total_samples
    }

    if visualize and output_dir and all_images:
        visualize_results_grid(
            all_images[:16], all_preds[:16], all_targets[:16],
            save_path=output_dir / "results_grid.png"
        )

    return metrics


def evaluate(config, args=None, checkpoint_path=None, output_dir=None, visualize=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    test_loader = data_loaders(split='test', return_trimaps=True, shuffle=False)

    model = create_segmentation_model(backbone=args.backbone)
    model = model.to(device)

    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        logger.warning("No checkpoint path provided. Using random weights.")

    vis_dir = output_dir / "visualizations"
    if visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_model(model, test_loader, device, visualize=visualize, output_dir=vis_dir)

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Evaluation completed. Metrics: {metrics}")
    return metrics
