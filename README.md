# Breast Ultrasound Image Segmentation using UNet++ Ensemble

This repository contains the official PyTorch implementation of the breast ultrasound image segmentation pipeline, utilizing a 5-fold ensemble of UNet++ architectures with EfficientNet-B4 encoders. 

## Overview
The primary objective of this project is to provide a robust and highly accurate segmentation model for breast ultrasound images. The pipeline compares classical image processing methods (e.g., Watershed, Region Growing) with a deep learning-based approach. To enhance generalization, the deep learning pipeline employs a 5-fold cross-validation strategy, and the final predictions are aggregated using Test-Time Augmentation (TTA).

## Dataset
The model is trained and evaluated on the **Breast Ultrasound Images (BUSI)** dataset. The pipeline handles data preparation, stratification, and validation strictly without data leakage.

## Computational Cost Analysis
To demonstrate the practical feasibility of our approach for computer-aided diagnosis (CAD) systems, inference and computational load were explicitly evaluated on an NVIDIA T4 GPU:

* **Model Complexity:** A single UNet++ (EfficientNet-B4) model contains **3.39 Million parameters**.
* **Computational Load:** A single forward pass requires **15.84 GFLOPs** (for a 256x256 input).
* **Inference Speed:**
  * Single Model Inference Time: **61.39 ms**
  * Full Ensemble Pipeline (5 models x 4 TTA configurations = 20 forward passes): **1227.76 ms**
  * System Speed: **0.81 FPS**

These metrics confirm that the proposed ensemble method operates efficiently while preserving high segmentation accuracy.

## Usage
The entire workflow, including data downloading, preprocessing, classical method evaluations, deep learning training, and statistical comparisons, is encapsulated within a single Jupyter Notebook. 

1. Open `BUSI_Segmentation_Pipeline.ipynb` in Google Colab.
2. Provide your Kaggle API token (`kaggle.json`) when prompted to download the BUSI dataset.
3. Run the cells sequentially. Checkpoints and outputs will be automatically saved to your mounted Google Drive.
