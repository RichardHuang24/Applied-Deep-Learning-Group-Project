import argparse
from handlers.classifier import handle_train_classifier
from handlers.masks import handle_generate_masks

def main():
    parser = argparse.ArgumentParser(description="WSSS Experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- Download (Optional)
    # ...

    # -- Train Classifier
    parser_classifier = subparsers.add_parser("train_classifier", help="Train image classifier")
    parser_classifier.add_argument("--config_path", required=True)
    parser_classifier.add_argument("--backbone", required=True)
    parser_classifier.add_argument("--init", required=True)
    parser_classifier.set_defaults(func=handle_train_classifier)


    # -- Generate Masks
    parser_masks = subparsers.add_parser("generate_masks", help="Generate CAM-based pseudo masks")
    parser_masks.add_argument("--config_path", required=True)
    parser_masks.add_argument("--cam", required=True, choices=["gradcam", "cam"])
    parser_masks.add_argument("--model-path", required=True)
    parser_masks.add_argument("--backbone", required=True)
    parser_masks.add_argument("--init", required=True)
    parser_masks.set_defaults(func=handle_generate_masks)

    # -- Train Classifier and Generate Masks
    parser_combo = subparsers.add_parser("train_and_generate", help="Train classifier and generate masks")
    parser_combo.add_argument("--config_path", required=True)
    parser_combo.add_argument("--backbone", required=True)
    parser_combo.add_argument("--init", required=True)
    parser_combo.add_argument("--cam", required=True, choices=["gradcam", "cam"])
    parser_combo.set_defaults(func=train_and_generate_masks)

    parser_classifier.add_argument("--experiment-name", default=None)
    parser_masks.add_argument("--experiment-name", required=True)
    parser_combo.add_argument("--experiment-name", default=None)

    # # -- Train Segmentation
    # parser_seg = subparsers.add_parser("train_segmentation", help="Train segmentation model")
    # parser_seg.add_argument("--config", required=True)
    # parser_seg.add_argument("--supervision", required=True)
    # parser_seg.add_argument("--pseudo-masks", required=True)
    # parser_seg.add_argument("--experiment", required=True)
    # parser_seg.set_defaults(func=handle_train_segmentation)

    # # -- Evaluate
    # parser_eval = subparsers.add_parser("evaluate", help="Evaluate segmentation model")
    # parser_eval.add_argument("--config", required=True)
    # parser_eval.add_argument("--supervision", required=True)
    # parser_eval.add_argument("--checkpoint", required=True)
    # parser_eval.add_argument("--experiment", required=True)
    # parser_eval.add_argument("--visualize", action="store_true")
    # parser_eval.set_defaults(func=handle_evaluate)

    # -- Run all in series
    parser_all = subparsers.add_parser("run_all", help="Run the full WSSS pipeline")
    parser_all.add_argument("--config", required=True)
    parser_all.add_argument("--backbone", required=True)
    parser_all.add_argument("--init", required=True)
    parser_all.add_argument("--cam", required=True, choices=["gradcam", "cam"])
    parser_all.set_defaults(func=handle_run_all)

    # Run selected subcommand
    args = parser.parse_args()
    args.func(args)

def train_and_generate_masks(args):
    """
    Trains the classifier and generates masks using the trained model.
    """
    # Step 1: Train classifier
    model_path, exp_name = handle_train_classifier(args)
    args.experiment_name = exp_name  # Inject for generate step

    # Step 2: Generate masks
    handle_generate_masks(args, model_path)

# Optional: Series runner
def handle_run_all(args):
    """
    Runs the full pipeline: classifier → CAM → masks → segmentation → evaluation
    """
    import time
    from argparse import Namespace

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment = f"{args.backbone}_{args.init}_{args.cam}_{timestamp}"

    # Step 1: Train classifier
    handle_train_classifier(Namespace(
        config=args.config,
        backbone=args.backbone,
        init=args.init,
        experiment=experiment
    ))

    # Step 2: Optionally train CAM model (if using CAM/CCAM)
    if args.cam == "cam":
        # handle_train_cam(Namespace(
        #     config=args.config,
        #     backbone=args.backbone,
        #     cam=args.cam,
        #     experiment=experiment
        # ))
        pass
        model_path = f"outputs/experiments/{experiment}/cam_model/best_model.pth"
    else:
        model_path = f"outputs/experiments/{experiment}/classifier/best_model.pth"

    # Step 3: Generate masks
    handle_generate_masks(Namespace(
        config=args.config,
        cam=args.cam,
        model_path=model_path,
        backbone=args.backbone,
        init=args.init,
        experiment=experiment,
        visualize=True
    ))

    # Step 4: Train segmentation
    # handle_train_segmentation(Namespace(
    #     config=args.config,
    #     supervision=f"weak_{args.cam}",
    #     pseudo_masks=f"outputs/experiments/{experiment}/masks",
    #     experiment=experiment
    # ))
    #
    # # Step 5: Evaluate
    # handle_evaluate(Namespace(
    #     config=args.config,
    #     supervision=f"weak_{args.cam}",
    #     checkpoint=f"outputs/experiments/{experiment}/segmentation/best_model.pth",
    #     experiment=experiment,
    #     visualize=True
    # ))

if __name__ == "__main__":
    main()
