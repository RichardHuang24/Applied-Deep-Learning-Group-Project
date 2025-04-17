# 🐾 WSSS Framework

A **Weakly-Supervised Semantic Segmentation** framework for training and evaluating segmentation models using CAM-based pseudo-labels and partial supervision.

## 📌 Overview

This framework implements a full pipeline for Weakly-Supervised Semantic Segmentation:

1. Training an image classifier with various backbones and initialization methods.
2. Generating Class Activation Maps (CAMs) using different techniques (e.g., GradCAM, CAM).
3. Creating pseudo-masks from the activation maps.
4. Training a segmentation model using the pseudo-masks or full masks.
5. Evaluating the segmentation performance with standard metrics.

---

## ⚙️ Installation

### Requirements

- Python 3.6+
- PyTorch 1.7+
- CUDA (recommended for faster training)
- **Additional dependencies**:  
  - `torch`  
  - `tqdm`  
  - `Pillow`

---

### 🔧 Setup

#### Clone the repository

```bash
git clone https://github.com/RichardHuang24/Applied-Deep-Learning-Group-Project.git
cd Applied-Deep-Learning-Group-Project
```

#### Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset

This framework is configured to work with the **Oxford-IIIT Pet Dataset**.

To download the dataset, run:

```bash
python main.py download
```

---

## 🚀 Running Experiments

### Run All Experiments

To run all experiment combinations with a specific configuration:

```bash
python main.py run_all --init random --cam gradcam --supervision weak_gradcam
```

---

## ⚙️ Customization Options

| Option        | Values                          | Description                                |
|---------------|----------------------------------|--------------------------------------------|
| `--init`      | `random`, `simclr`, `imagenet`   | Initialization method                      |
| `--cam`       | `gradcam`, `cam`                 | CAM method                                 |
| `--supervision` | `weak_gradcam`, `weak_cam`, `full` | Type of supervision                       |

---

## 📦 Understanding the Output

For each experiment, the following outputs are generated:

1. **Classifier Model**: Trained image classifier
2. **CAM Model**: Class Activation Map generator
3. **Generated Masks**: Pseudo-masks from CAMs
4. **Segmentation Model**: Final segmentation model (e.g., PSPNet)
5. **Evaluation Results**: mIoU and pixel accuracy scores
6. **Experiment Summary**: Visual and JSON summary

#### Output Directory Structure:

```
experiments/
└── resnet50_imagenet_gradcam_20230415_120000/
    ├── classifier/
    ├── cam_model/
    ├── masks/
    │   └── visualizations/
    ├── segmentation/
    ├── evaluation/
    ├── experiment_config.json
    ├── results.json
    ├── experiment_summary.png
    └── experiment.log
```

---

## 📊 Metrics

The framework evaluates segmentation performance using:

- **Pixel Accuracy**: Percentage of correctly classified pixels.
- **Mean IoU (mIoU)**: Average Intersection over Union across all classes.

All results are automatically saved to `results/metrics_table.csv`.

---

## 🧪 Experiment Workflow Example

```bash
# 1. Download the dataset
python main.py download

# 2. Run a fast test experiment
python main.py run_all --init random --cam gradcam --supervision weak_gradcam

# 3. Run a high-performance configuration
python main.py run_all --init imagenet --cam cam --supervision weak_cam

# 4. Run a fully supervised experiment
python main.py run_all --init simclr --cam gradcam --supervision full
```

---

## 📂 Project Structure

```
├── main.py                  # Main runner
├── train.py                 # Training pipeline
├── generate_masks.py        # CAM mask generation
├── evaluate.py              # Evaluation utilities
├── utils/                   # Utility functions
│   └── download.py          # Dataset utils
├── models/
│   ├── classifier/          # ResNet-based classifiers
│   ├── cam/                 # CAM methods: GradCAM, CAM
│   └── segmentation/        # PSPNet (semantic segmentation)
├── data/                    # Dataset handling
├── config.json              # Default experiment configuration
└── requirements.txt         # Dependencies
```

---

## 📜 Citing

If you use this framework in your research or coursework, please cite this repository:

> GitHub: [RichardHuang24/Applied-Deep-Learning-Group-Project](https://github.com/RichardHuang24/Applied-Deep-Learning-Group-Project)

---

Happy segmenting! 🎯
