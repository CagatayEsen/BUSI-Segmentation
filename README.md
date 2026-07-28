# Breast Ultrasound Image Segmentation: Classical vs. Deep Learning Ensemble

This repository contains the official PyTorch implementation of a comprehensive breast ultrasound image segmentation pipeline. The project compares classical image processing techniques with a robust deep learning ensemble approach.

## Dataset and Preprocessing
The model is developed and evaluated using the **Breast Ultrasound Images (BUSI)** dataset. 

* **Data Preparation:** The pipeline strictly prevents data leakage by implementing robust train/validation/test splits. 
* **Preprocessing:** Raw ultrasound images often contain artifacts such as text and medical markers. An OCR-based text removal algorithm is applied to clean the images before processing. Additionally, for images containing multiple tumor masks, the masks are merged into a single binary ground truth mask using a logical OR operation. Normal (tumor-free) images are also included in the training and evaluation phases.

## Methodology
The project explores two distinct approaches to medical image segmentation:

### 1. Classical Segmentation
As a baseline, traditional image processing algorithms are implemented and evaluated without deep learning interventions. The classical pipeline relies on:
* Otsu's Thresholding
* Watershed Algorithm
* Region Growing

### 2. Deep Learning Ensemble
The primary and most accurate approach relies on deep convolutional neural networks:
* **Architecture:** UNet++ equipped with an EfficientNet-B4 encoder.
* **Validation Strategy:** A strict 5-fold cross-validation protocol is used to ensure the model's generalizability and to prevent overfitting.
* **Ensemble and TTA:** At inference, the predictions from the 5 independent fold models are aggregated. Each model's output is further stabilized using Test-Time Augmentation (TTA), specifically averaging over four flip configurations (identity, vertical, horizontal, and both axes). The final combined probability map is thresholded at 0.5 to produce the ultimate segmentation mask.

## Usage
The entire workflow, from data downloading to classical method evaluations, deep learning training, and statistical comparisons, is encapsulated within a single Jupyter Notebook.

1. Open `BUSI_Segmentation_Pipeline.ipynb` in Google Colab.
2. Provide your Kaggle API token (`kaggle.json`) when prompted to download the BUSI dataset automatically.
3. Run the cells sequentially. Model weights (checkpoints) and evaluation outputs are configured to save directly to your mounted Google Drive.
