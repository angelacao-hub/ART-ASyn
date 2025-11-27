# ART-ASyn: Anatomy-aware Realistic Texture-based Anomaly Synthesis Framework for Chest X-Rays

## Overview

ART-ASyn presents an anatomy-aware realistic texture-based framework for synthesizing medical anomalies in chest X-ray images. The method generates realistic synthetic anomalies that preserve anatomical consistency and texture patterns for improved anomaly detection and classification in medical imaging.

## Data Acquisition

All datasets mentioned in the paper are available under the `data/` directory:
- **CheXpert**: Located at `data/CheXpert/`
- **ZhangLab**: Located at `data/ZhangLab/`  
- **QaTa-ZeroShot**: Located at `data/QaTa-ZeroShot/`

Please refer to the respective dataset documentation for acquisition details of each dataset.

## Usage

### CheXpert Dataset
```bash
# Preprocessing
cd ART-ASyn/preprocessing
python ART_ASyn.py --dataset CheXpert

# Training
cd ART-ASyn
python train.py --dataset CheXpert --data_dir ../data/CheXpert

# Testing
cd ART-ASyn
python test.py --dataset CheXpert --data_dir ../data/CheXpert --model_path <saved_model_path>
```

### ZhangLab Dataset
```bash
# Preprocessing
cd ART-ASyn/preprocessing
python preprocessing/ART_ASyn.py --dataset ZhangLab

# Training
cd ART-ASyn
python train.py --dataset ZhangLab --data_dir ../data/ZhangLab

# Testing
cd ART-ASyn
python test.py --dataset ZhangLab --data_dir ../data/ZhangLab --model_path <saved_model_path>
```

### QaTa-ZeroShot Dataset
```bash
# Preprocessing
cd ART-ASyn/preprocessing
python preprocessing/ART_ASyn.py --dataset QaTa-ZeroShot

# Training
cd ART-ASyn
python train.py --dataset QaTa-ZeroShot --data_dir ../data/QaTa-ZeroShot

# Testing
cd ART-ASyn
python test.py --dataset QaTa-ZeroShot --data_dir ../data/QaTa-ZeroShot --model_path <saved_model_path>
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