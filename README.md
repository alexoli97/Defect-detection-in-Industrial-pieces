# Fraunhofer Project Documentation

## Overview
This document outlines the methodologies, tools, and outcomes of the Fraunhofer project. The primary focus of the project was to develop a comprehensive solution for image inspection, utilizing various image processing and machine learning techniques. The work involved the creation and analysis of image datasets, implementation of models for defect detection and classification, and the documentation of results.

## Repository Structure
The project repository consists of several key components, detailed as follows:

### 1. InspectionTry Directory
- **Purpose:** Created as a primary workspace to organize images and masks for efficient processing and analysis.
- **Contents:** 
  - `Gear` folder: Contains real and synthetic images, masks, and subfolders for gear images, including a 'labels' folder with annotated gear images.
  - `Spring` folder: Similar to the Gear folder, it includes real and synthetic images and masks for spring images.
  - `Blender_Synthetic_Images`: Contains images created in Blender for both gear and spring datasets.
  - `Classification`: Includes labeled images for defect classification, divided into Gear and Spring subfolders.
  - `Input Images`: A set of 8 test images (2 synthetic and 2 real for both gear and spring) for model validation.
  - `Prediction Input Image`: Contains model outputs using the 'Input Images'.
  - Python scripts and pretrained model weights (excluding `DefectSegmentationUnet.py` and `Defect_classification.ipynb`).

### 2. Scripts in GitHub
- **DefectSegmentationUnet.py**: A Unet CNN model for defect segmentation, with specific implementations for gear and spring images.
- **simple_unet_model.py**: Defines the Unet model architecture.
- **data_augmentation.py**: Enhances the dataset size for better model training.
- **Gabor.train.py and Gabor.predict.py**: Implements a Random Forest model with image filters, particularly Gabor filters, for defect segmentation.
- **classification.py**: Classifies images as either gear or spring and applies appropriate segmentation models.
- **Defect_classification.py**: A CNN+RF model for defect type prediction, with limited success.
- **Multi_classification.py**: Employs a U-net model for defect type prediction.
- **Defect_classification.ipynb**: Experiments with complex deep learning models using various backbones (ResNet34, InceptionV3, VGG16) for defect classification.

## Results and Observations
- The defect segmentation models showed promising results for gear images, particularly with synthetic data. However, the complexity of spring images posed challenges in achieving similar accuracy.
- The defect classification models were less effective, especially in distinguishing different types of defects.
- The project successfully demonstrated the application of machine learning techniques in image inspection, despite some limitations in specific areas.

## Tools and Environment
- **Development Environments**: Spyder (Anaconda TensorFlow) and Google Colab.
- **Languages and Frameworks**: Python, TensorFlow.

## Conclusion
The Fraunhofer project achieved considerable success in image processing and defect detection using machine learning models. While certain aspects, such as spring image segmentation and defect classification, require further refinement, the project lays a solid foundation for future enhancements in automated image inspection technologies.
