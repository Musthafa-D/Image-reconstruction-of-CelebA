# CelebA Image Reconstruction (PyTorch)

## Project Overview

This repository demonstrates image reconstruction on the **CelebA** dataset using deep learning models implemented in PyTorch.  
Image reconstruction focuses on learning to represent an input image in a compressed form and then reconstruct it as accurately as possible. This is a foundational task in representation learning.

The CelebA dataset consists of large facial image data with multiple labeled attributes, but this project focuses exclusively on reconstruction quality, not classification.

This project focuses on training a reconstruction model to generate outputs that are visually similar to the original CelebA images and then interpret the trained model using various interpretability methods such as atrributions, metrics, etc.

---

## Dataset

- **CelebA (Celeb Faces Attributes)**
  - ~200,000 face images
  - Variety of facial attributes such as smiling, eyeglasses, hairstyles
  - Images are typically 178×218 and cropped/resized for model input

Dataset preparation is typically required due to its size (download from official source and preprocess before training).

---

## What This Project Includes

- Loading and preprocessing the CelebA dataset using PyTorch
- Deep learning model architecture suitable for image reconstruction
- Training loop to minimize reconstruction loss
- Evaluation and visualization of reconstructed images
- Saving sample outputs during training
- Interpreting the trained models.

---
