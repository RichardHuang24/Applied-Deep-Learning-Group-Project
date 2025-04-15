"""
Main execution script for WSSS experiments
"""
import os
import torch
import json
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from PIL import Image, ImageDraw, ImageFont

from train import train_classifier, train_cam, train_segmentation
from generate_masks import generate_masks
from evaluate import evaluate, evaluate_all_models
from utils.download import download_dataset

logger = logging.getLogger(__name__)

def setup_logging(output_dir):
    """Set up logging to file and console"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / "experiment.log"),
            logging.StreamHandler()
        ]
    )

def run_experiment(config_path, backbone, initialization, cam_method, output_base_dir=None):
    """
    Run a complete experiment with specified configuration
    
    Args:
        config_path: Path to configuration file
        backbone: Backbone model name ('resnet18', 'resnet34', 'resnet50')
        initialization: Initialization method ('simclr', 'imagenet', 'random')
        cam_method: CAM method ('gradcam', 'ccam')
        output_base_dir: Base directory for outputs
        
    Returns:
        dict: Dictionary with experiment results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set up output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{backbone}_{initialization}_{cam_method}_{timestamp}"
    
    if output_base_dir is None:
        output_base_dir = Path(config['paths']['outputs'])
    else:
        output_base_dir = Path(output_base_dir)
    
    output_dir = output_base_dir / "experiments" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Backbone: {backbone}, Initialization: {initialization}, CAM: {cam_method}")
    
    experiment_config = {
        'backbone': backbone,
        'initialization': initialization,
        'cam_method': cam_method
    }
    
    # Save experiment config
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(experiment_config, f, indent=4)
    
    # Step 1: Train classifier
    logger.info("Step 1: Training classifier")
    classifier_name = f"{backbone}_{initialization}"
    classifier_path = train_classifier(
        config_path=config_path,
        experiment=classifier_name,
        output_dir=output_dir / "classifier"
    )
    
    # Step 2: Train CAM model if needed (especially for CCAM)
    logger.info(f"Step 2: Training/Preparing {cam_method} model")
    if cam_method == 'ccam':  # CCAM requires explicit training
        cam_model_path = train_cam(
            config_path=config_path,
            method=cam_method,
            backbone=backbone,
            classifier_path=classifier_path,
            output_dir=output_dir / "cam_model"
        )
    else:
        cam_model_path = classifier_path  # GradCAM uses the classifier directly
    
    # Step 3: Generate masks using CAM
    logger.info(f"Step 3: Generating masks using {cam_method}")
    masks_dir = generate_masks(
        config_path=config_path,
        method=cam_method,
        classifier_path=cam_model_path,
        output_dir=output_dir / "masks",
        visualize=True
    )
    
    # Step 4: Train segmentation model
    logger.info("Step 4: Training segmentation model")
    supervision = f"weak_{cam_method}"
    segmentation_path = train_segmentation(
        config_path=config_path,
        supervision=supervision,
        pseudo_masks_dir=masks_dir,
        output_dir=output_dir / "segmentation"
    )
    
    # Step 5: Evaluate segmentation model
    logger.info("Step 5: Evaluating segmentation model")
    metrics = evaluate(
        config_path=config_path,
        supervision=supervision,
        checkpoint_path=segmentation_path,
        output_dir=output_dir / "evaluation",
        visualize=True
    )
    
    # Save results
    results = {
        'experiment': experiment_name,
        'backbone': backbone,
        'initialization': initialization,
        'cam_method': cam_method,
        'metrics': metrics
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Experiment completed. Results saved to {output_dir}")
    
    # Create a simple visualization summary
    create_experiment_summary(output_dir, results)
    
    return results

def create_experiment_summary(output_dir, results):
    """Create a visual summary of experiment results"""
    # Find sample images
    eval_dir = output_dir / "evaluation"
    mask_dir = output_dir / "masks" / "visualizations"
    
    sample_images = []
    
    # Get a sample from evaluation results
    eval_samples = list(eval_dir.glob("*_pred.png"))
    if eval_samples:
        sample_images.append(("Segmentation Result", eval_samples[0]))
    
    # Get a sample from mask visualizations
    mask_samples = list(mask_dir.glob("*.png"))
    if mask_samples:
        sample_images.append(("CAM Visualization", mask_samples[0]))
    
    # Create summary image
    width, height = 800, 600
    summary = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(summary)
    
    # Draw title
    exp_name = f"{results['backbone']} + {results['initialization']} + {results['cam_method']}"
    draw.text((20, 20), f"Experiment Summary: {exp_name}", fill='black')
    
    # Draw metrics
    metrics_text = [
        f"Pixel Accuracy: {results['metrics']['pixel_acc']:.4f}",
        f"Mean IoU: {results['metrics']['miou']:.4f}"
    ]
    
    for i, text in enumerate(metrics_text):
        draw.text((20, 60 + i*30), text, fill='black')
    
    # Add sample images
    y_offset = 150
    for title, img_path in sample_images:
        try:
            img = Image.open(img_path)
            # Resize while maintaining aspect ratio
            img.thumbnail((width - 40, 200))
            summary.paste(img, (20, y_offset))
            draw.text((20, y_offset - 20), title, fill='black')
            y_offset += img.height + 40
        except Exception as e:
            logger.error(f"Error adding sample image: {e}")
    
    # Save summary
    summary.save(output_dir / "experiment_summary.png")

def run_all_experiments(config_path, output_base_dir=None):
    """
    Run all combinations of experiments
    
    Args:
        config_path: Path to configuration file
        output_base_dir: Base directory for outputs
        
    Returns:
        dict: Dictionary with all results
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set up output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if output_base_dir is None:
        output_base_dir = Path(config['paths']['outputs'])
    else:
        output_base_dir = Path(output_base_dir)
    
    output_dir = output_base_dir / "all_experiments" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    
    logger.info("Running all experiment combinations")
    
    # Get configurations
    backbones = config['models']['classifier']['backbones']
    initializations = config['models']['classifier']['initializations']
    cam_methods = config['models']['cam']['methods']
    
    # Store all results
    all_results = {}
    
    # Run all combinations
    total_experiments = len(backbones) * len(initializations) * len(cam_methods)
    with tqdm(total=total_experiments, desc="Running experiments") as pbar:
        for backbone in backbones:
            for initialization in initializations:
                for cam_method in cam_methods:
                    exp_name = f"{backbone}_{initialization}_{cam_method}"
                    logger.info(f"Running experiment: {exp_name}")
                    
                    try:
                        results = run_experiment(
                            config_path=config_path,
                            backbone=backbone,
                            initialization=initialization,
                            cam_method=cam_method,
                            output_base_dir=output_dir / exp_name
                        )
                        all_results[exp_name] = results
                    except Exception as e:
                        logger.error(f"Error in experiment {exp_name}: {e}")
                    
                    pbar.update(1)
    
    # Save all results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Run comprehensive evaluation comparing all models
    logger.info("Generating comprehensive model comparison...")
    compare_results = evaluate_all_models(
        config_path=config_path,
        output_dir=output_dir / "comparison"
    )
    
    # Create results table for visualization
    create_results_table(output_dir, all_results, backbones, initializations, cam_methods)
    
    logger.info(f"All experiments completed. Results saved to {output_dir}")
    logger.info(f"Comprehensive comparison available at {output_dir / 'comparison'}")
    
    return all_results

def create_results_table(output_dir, all_results, backbones, initializations, cam_methods):
    """Create a visual table of all experiment results"""
    # Setup dimensions
    cell_width, cell_height = 150, 80
    header_height = 60
    width = cell_width * (len(initializations) + 1)
    height = header_height + cell_height * (len(backbones) * len(cam_methods) + 1)
    
    # Create image and drawing context
    table = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(table)
    
    # Draw grid and headers
    draw_grid_lines(draw, width, height, cell_width, cell_height, header_height, 
                   len(initializations), len(backbones), len(cam_methods))
    draw_headers(draw, initializations, cell_width, header_height)
    
    # Fill results
    draw_results(draw, all_results, backbones, initializations, cam_methods, 
                cell_width, cell_height, header_height)
    
    # Save table
    table_path = output_dir / "results_table.png"
    table.save(table_path)
    logger.info(f"Results table saved to {table_path}")
    
def draw_grid_lines(draw, width, height, cell_width, cell_height, header_height, 
                   num_init, num_backbones, num_cam):
    """Draw grid lines for the table"""
    # Draw vertical lines
    for i in range(num_init + 2):
        x = i * cell_width
        draw.line([(x, 0), (x, height)], fill='black', width=2)
    
    # Draw horizontal lines
    for j in range(num_backbones * num_cam + 3):
        y = j * cell_height if j > 0 else header_height
        draw.line([(0, y), (width, y)], fill='black', width=2)

def draw_headers(draw, initializations, cell_width, header_height):
    """Draw column and row headers"""
    # Draw model header
    draw.text((20, 20), "Model", fill='black')
    
    # Draw initialization headers
    for i, init in enumerate(initializations):
        x = (i + 1) * cell_width + 20
        draw.text((x, 20), init, fill='black')

def draw_results(draw, all_results, backbones, initializations, cam_methods, 
                cell_width, cell_height, header_height):
    """Draw result cells with color coding"""
    row = 1
    
    for backbone in backbones:
        for cam_method in cam_methods:
            # Draw row header
            y = header_height + (row - 1) * cell_height + 20
            draw.text((20, y), f"{backbone}\n{cam_method}", fill='black')
            
            # Draw results for each initialization
            for i, initialization in enumerate(initializations):
                exp_name = f"{backbone}_{initialization}_{cam_method}"
                
                if exp_name in all_results:
                    draw_result_cell(draw, all_results[exp_name], i, row, 
                                   cell_width, cell_height, header_height)
            
            row += 1

def draw_result_cell(draw, result, col_idx, row_idx, cell_width, cell_height, header_height):
    """Draw individual result cell with color coding"""
    pixel_acc = result['metrics']['pixel_acc']
    miou = result['metrics']['miou']
    
    # Calculate position
    x = (col_idx + 1) * cell_width + 20
    y = header_height + (row_idx - 1) * cell_height + 20
    
    # Color code based on mIoU
    color_val = int(255 * miou)
    cell_color = (255, 255 - color_val, 255 - color_val)  # Red to white gradient
    
    # Draw colored background
    x1 = col_idx * cell_width + cell_width
    y1 = header_height + (row_idx - 1) * cell_height
    x2 = x1 + cell_width
    y2 = y1 + cell_height
    draw.rectangle([x1, y1, x2, y2], fill=cell_color, outline="black")
    
    # Draw metrics text
    draw.text((x, y), f"PA: {pixel_acc:.2f}\nmIoU: {miou:.2f}", fill='black')

def main():
    """Main entry point for the experiment runner"""
    parser = argparse.ArgumentParser(description="Run WSSS experiments")
    
    parser.add_argument(
        "--config", 
        default="config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--backbone", 
        choices=["resnet18", "resnet34", "resnet50"],
        default="resnet50",
        help="Backbone architecture"
    )
    
    parser.add_argument(
        "--init", 
        choices=["simclr", "imagenet", "random"],
        default="random",
        help="Initialization method"
    )
    
    parser.add_argument(
        "--cam", 
        choices=["gradcam", "ccam"],
        default="gradcam",
        help="CAM method"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all combinations of experiments"
    )
    
    parser.add_argument(
        "--download", 
        action="store_true",
        help="Download dataset before running experiments"
    )
    
    parser.add_argument(
        "--download-only", 
        action="store_true",
        help="Only download dataset without running experiments"
    )
    
    parser.add_argument(
        "--output", 
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
    
    # Download dataset if requested
    if args.download or args.download_only:
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        dataset_dir = Path(config['dataset']['root'])
        print(f"Downloading dataset to {dataset_dir}")
        download_dataset(dataset_dir)
        
        # Exit if only downloading was requested
        if args.download_only:
            print("Dataset download complete. Exiting without running experiments.")
            return
    
    # Run experiments
    if args.all:
        print("Running all experiment combinations")
        run_all_experiments(
            config_path=args.config,
            output_base_dir=args.output
        )
    else:
        print(f"Running single experiment: {args.backbone} + {args.init} + {args.cam}")
        run_experiment(
            config_path=args.config,
            backbone=args.backbone,
            initialization=args.init,
            cam_method=args.cam,
            output_base_dir=args.output
        )

if __name__ == "__main__":
    main()