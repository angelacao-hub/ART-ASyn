# ART-ASyn: Anatomy-aware Realistic Texture-based Anomaly Synthesis Framework for Chest X-Rays

## Overview
This repository contains the official implementation for the WACV 2026 accepted paper **"ART-ASyn: Anatomy-aware Realistic Texture-based Anomaly Synthesis Framework for Chest X-Rays"**.

<center>
  <img src="ART-ASyn Diagram.svg" width="800" alt="ART-ASyn Diagram">
</center>

Our work  presents a novel **A**natomy-aware **R**ealistic **T**exture-based **A**nomaly **Syn**thesis framework (**ART-ASyn**) for chest X-rays that generates realistic and anatomically consistent lung opacity related anomalies using texture-based augmentation guided by our proposed **P**rogressive **B**inary **T**hresholding **Seg**mentation method (**PBTSeg**) for lung segmentation.

<center>
  <img src="PBTSeg Diagram.svg" width="70%" alt="PBTSeg Diagram">
</center>

## Data Format & Acquisition

The framework expects data to be organized in the following directory structure:

```
data/
└── <dataset_name>/
    ├── train/
    │   └── healthy/
    └── test/
        ├── healthy/
        └── diseased/
```

All datasets mentioned in the paper are available under the `data/` directory:
- **CheXpert**: Located at `data/CheXpert/`
- **ZhangLab**: Located at `data/ZhangLab/`  
- **QaTa-ZeroShot**: Located at `data/QaTa-ZeroShot/`

Please refer to the respective dataset documentation for acquisition details of each dataset.

## Usage

#### Preprocessing
Before running the framework, ensure the following preprocessing steps are completed:

**Step 1: PBTSeg Lung Segmentation**
```bash
cd ART-ASyn/preprocessing
python PBTSeg.py --dataset <dataset_name>
```
This step performs progressive binary thresholding to generate accurate lung masks for anatomical constraint.

**Step 2: ART-ASyn Anomaly Synthesis**
```bash
cd ART-ASyn/preprocessing  
python ART_ASyn.py --dataset <dataset_name>
```
This step generates realistic synthetic anomalies with pixel-level masks using texture-based augmentation.

#### Training
After preprocessing, train the model:
```bash
cd ART-ASyn
python train.py --dataset <dataset_name> --data_dir ../data/<dataset_name>
```

#### Testing
Evaluate the trained model:
```bash
cd ART-ASyn
python test.py --dataset <dataset_name> --data_dir ../data/<dataset_name> --model_path <saved_model_path>
```

#### Complete Workflow Example - CheXpert
```bash
# Complete preprocessing pipeline
cd ART-ASyn/preprocessing
python PBTSeg.py --dataset CheXpert
python ART_ASyn.py --dataset CheXpert

# Training
cd ../../
python train.py --dataset CheXpert --data_dir data/CheXpert

# Testing
python test.py --dataset CheXpert --data_dir data/CheXpert --model_path <saved_model_path>
```

### Python Environment
- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux

### Dependencies Installation
All required packages are listed in `requirements.txt`. Install using:

```bash
pip install -r requirements.txt
```

## Citation

```bibtex
@inproceedings{art_asyn_2026,
  title={ART-ASyn: Anatomy-aware Realistic Texture-based Anomaly Synthesis Framework for Chest X-Rays},
  author={Qinyi Cao, Jianan Fan, Weidong Cai},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026},
  year={2026}
}
```