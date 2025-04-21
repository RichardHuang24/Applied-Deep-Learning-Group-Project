# 🐾 WSSS Framework

A **Weakly-Supervised Semantic Segmentation** framework for training and evaluating segmentation models using CAM-based pseudo-labels and partial or full supervision.

---

## 📌 Overview

This framework implements a modular pipeline for weakly-supervised semantic segmentation using the Oxford-IIIT Pet dataset. It supports:
- Training classifiers with various backbones and initialization types
- Generating Class Activation Maps (CAMs)
- Converting CAMs to pseudo segmentation masks
- Training segmentation models using weak or full supervision
- Evaluation and result aggregation

All components are accessible via command-line interfaces in `main.py`.

---

## ⚙️ Installation

### Requirements

- Python 3.6+
- PyTorch 1.7+
- CUDA (recommended for faster training)
- **Additional dependencies**:  
  - `tqdm`  

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

---

## 📁 Dataset

This framework is configured to work with the **Oxford-IIIT Pet Dataset**.

To download the dataset, run:

```bash
python main.py download
```

---

## 🚀 Running Experiments

All commands below are available via `main.py`. You can run individual steps or the entire pipeline.

---

### 🔹 Step 1: Train a Classifier

```bash
python main.py train_classifier     --backbone resnet50     --init imagenet     --cam gradcam     --experiment_name example_exp
```

---

### 🔹 Step 2: Generate CAM-based Pseudo Masks

```bash
python main.py generate_masks     --backbone resnet50     --init imagenet     --cam cam+ccam     --model_path path/to/classifier.pth     --experiment_name example_exp
```

---

### 🔹 Step 3: Train the Segmentation Model

```bash
python main.py train_segmentation     --supervision weak_gradcam     --init imagenet     --cam gradcam     --experiment_name example_exp
```

---

### 🔹 Step 4: Evaluate Segmentation Performance

```bash
python main.py evaluate     --supervision weak_gradcam     --init imagenet     --cam gradcam     --experiment_name example_exp
```

---

### 🔹 Combined Step: Train Classifier + Generate CAM Masks

```bash
python main.py train_and_generate     --backbone resnet50     --init imagenet     --cam gradcam+ccam     --experiment_name example_exp
```

---

### 🔹 Run Full Pipeline

```bash
python main.py run_all     --backbone resnet50     --init imagenet     --cam gradcam+ccam     --supervision weak_gradcam     --experiment_name example_exp
```

---

## ⚙️ Customization Options

| Option         | Values                                             | Description                      |
|----------------|-----------------------------------------------------|----------------------------------|
| `--init`       | `random`, `simclr`, `imagenet`                     | Initialization method            |
| `--cam`        | `gradcam`, `cam`, `ccam`, `gradcam+ccam`, `cam+ccam` | CAM methods                      |
| `--supervision`| `full`, `weak_gradcam`, `weak_cam`                 | Supervision type                 |

---

## 📦 Understanding the Output

For each experiment, the following outputs are generated:

1. **Classifier Model**: Trained image classifier
2. **CAM Model**: Class Activation Map generator
3. **Generated Masks**: Pseudo-masks from CAMs
4. **Segmentation Model**: Final segmentation model (e.g., PSPNet)
5. **Evaluation Results**: mIoU and pixel accuracy scores
6. **Experiment Summary**: Visual and JSON summary

## 📦 Output Structure

```
outputs/
└── example_exp/
    ├── classifier/
    ├── masks/
    ├── segmentation/
    ├── evaluation/
    ├── experiment_config.json
    └── results.json
```

---

## 📊 Metrics

The framework evaluates segmentation performance using:

- **Pixel Accuracy**: Percentage of correctly classified pixels.
- **Mean IoU (mIoU)**: Average Intersection over Union across all classes.

All results are automatically saved to `outputs/experiments.log`.

---

## 📂 Project Structure

```
├── main.py
├── train.py
├── generate_masks.py
├── evaluate.py
├── data.py

├── handlers/
│   ├── classifier.py
│   ├── segmentation.py
│   ├── masks.py
│   └── evaluate.py

├── models/
│   ├── classifier.py
│   ├── cam.py
│   └── pspnet.py

├── utils/
│   ├── download.py
│   ├── metrics.py
│   ├── visualization.py
│   └── logging.py

├── config.json
├── requirements.txt
└── README.md
```


---

## 📜 Citation

> GitHub: [RichardHuang24/Applied-Deep-Learning-Group-Project](https://github.com/RichardHuang24/Applied-Deep-Learning-Group-Project)

