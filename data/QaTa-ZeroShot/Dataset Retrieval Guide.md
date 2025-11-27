# QaTa-ZeroShot Dataset Guide

## Dataset Overview

### **Dataset Name**: QaTa-ZeroShot Dataset
- **Purpose**: Zero-shot learning for COVID-19 detection in chest X-ray images
- **Architecture**: Train and validation sets from ZhangLab and CheXpert, test set from QaTa-Cov19-v2

## Dataset Construction

### **Data Sources and Composition**

#### **Training Data (Healthy Samples)**
The training set consists exclusively of healthy (non-COVID) chest X-ray images sourced from preprosessed medical imaging datasets:

##### **1. ZhangLab Healthy Samples**
- **Source Directory**: `data/ZhangLab/train/healthy/` (If not retrived, follow instructions in `data/ZhangLab/Data Preprocessing.ipynb`)
- **Copy ZhangLab Healthy Samples**:
   ```bash
   cp -r data/ZhangLab/train/healthy/* data/QaTa-ZeroShot/train/healthy/
   ```

##### **2. CheXpert Healthy Samples**
- **Source Directory**: `data/CheXpert/train/healthy/` (If not retrived, follow instructions in `data/CheXpert/Data Preprocessing.ipynb`)
- **Copy CheXpert Healthy Samples**:
   ```bash
   cp -r data/CheXpert/train/healthy/* data/QaTa-ZeroShot/train/healthy/
   ```

#### **Target Directory for Training Data**
- **Destination**: `data/QaTa-ZeroShot/train/healthy/`
- **Process**: Copy all healthy samples from both ZhangLab and CheXpert datasets

#### **Test Data (COVID-19 Samples)**
The test set contains COVID-19 positive chest X-ray images for evaluation purposes:

##### **QaTa-COV19 Dataset Source**
- **Kaggle URL**: [QaTa-COV19 Dataset](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)
- **Archive File**: `archive.zip`
- **Size**: Approximately 120K chest X-ray images in the full dataset
- **COVID-19 Cases**: 9,258 annotated COVID-19 chest X-ray images

##### **QaTa-COV19 Dataset Structure**
```
QaTa-Cov19/
├── archive.zip (to be extracted)
├── <other_folders>
└── QaTa-COV19-v2/
    ├── Ground-truths/    # Annotation files for COVID-19 lesions
    └── Images/          # COVID-19 chest X-ray images
```

1. **Navigate to QaTa-COV19-v2 directory**: `QaTa-COV19/QaTa-COV19-v2/`
2. **Copy Ground Truth Files**:
   - Source: `QaTa-COV19-v2/Ground-truths/`
   - Destination: `data/QaTa-ZeroShot/ground_truth/`
3. **Copy COVID-19 Images**:
   - Source: `QaTa-COV19-v2/Images/`
   - Destination: `data/QaTa-ZeroShot/test/diseased/`

### **Final Directory Structure**
After setup, your directory structure should look like:
```
data/QaTa-ZeroShot/
├── train/
│   └── healthy/
│       ├── [ZhangLab healthy samples]
│       └── [CheXpert healthy samples]
├── test/
│   └── diseased/
│       └── [QaTa-COV19 COVID-19 samples]
└── ground_truth/
    └── [COVID-19 annotation files]
```