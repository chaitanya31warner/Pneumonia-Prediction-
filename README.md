# Pneumonia-Prediction
Pneumonia Prediction from Chest X-ray Images: Comparative Analysis of Machine Learning and Deep Learning Models
Here is a sample `README.md` for your project based on the information from your document:

---

# Pneumonia Prediction from Chest X-ray Images: Comparative Analysis of Machine Learning and Deep Learning Models


## Introduction
Pneumonia is a critical public health issue, causing millions of deaths annually. This project investigates various machine learning (ML) and deep learning (DL) models, comparing their effectiveness in detecting pneumonia from chest X-ray images. The study includes traditional ML methods such as SVM, KNN, and Decision Trees, as well as advanced DL approaches, notably CNNs and transfer learning with VGG16 and ResNet50.

The goal is to identify the best performing model for accurate pneumonia detection.

## Dataset
The dataset consists of approximately 6,000 X-ray images of normal and pneumonia cases, sourced from publicly available datasets. It is split into training, testing, and validation sets.

- **Classes**: Normal, Pneumonia
- **Size**: ~2 GB

## Architecture and Methodology
### Feature Extraction
Feature extraction is performed using CNN models, which extract key information from the X-ray images through multiple convolutional and pooling layers. The extracted features are then used for training machine learning classifiers.

### CNN Architecture
The CNN model includes:
- Multiple convolutional layers (32, 64, 128 filters)
- Max-pooling layers
- Fully connected layers with ReLU activation
- Dropout for regularization
- Sigmoid activation in the output layer for binary classification

### Transfer Learning
Pre-trained models such as VGG16 and ResNet50 are fine-tuned to enhance pneumonia prediction. These models are chosen due to their proven ability in image recognition tasks.

## Models Implemented
1. **Convolutional Neural Network (CNN)**: Built from scratch for feature extraction.
2. **VGG16**: A pre-trained model fine-tuned for pneumonia detection.
3. **ResNet50**: Another pre-trained model adapted to our dataset.
4. **Traditional Machine Learning Models**:
   - SVM
   - K-Nearest Neighbors (KNN)
   - Decision Trees
   - Random Forest
   - AdaBoost

## Installation
### Requirements
- Python 3.x
- Libraries: 
  ```bash
  pip install -r requirements.txt
  ```

### Dependencies
```
tensorflow
keras
scikit-learn
opencv-python
numpy
pandas
matplotlib
seaborn
```

## Usage
### 1. Data Preprocessing
Preprocess the dataset by resizing and normalizing the images.
```python
from src.preprocessing import preprocess_data
train_data, test_data = preprocess_data('data/')
```

### 2. Train the Models
To train the deep learning models:
```bash
python src/train_cnn.py
python src/train_vgg16.py
```

For machine learning models:
```bash
python src/train_ml_models.py
```

### 3. Evaluate the Models
Run the evaluation script to compare the performance:
```bash
python src/evaluate_models.py
```

## Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC AUC**

## Results
- **VGG16**: Accuracy of 88.86%
- **ResNet50**: Accuracy of 84.20%
- **CNN**: Accuracy of 70%
- **Random Forest**: Accuracy of 64.26%
- **AdaBoost**: Accuracy of 63.76%

The comparison clearly shows that deep learning models outperform traditional machine learning models in pneumonia detection.

## Conclusion and Future Work
This study demonstrated that deep learning models, especially VGG16, achieve higher accuracy in pneumonia detection from chest X-ray images. In future work, we aim to improve the model's interpretability and explore multi-class classification tasks.

## References
1. Ayan, E., & Ünver, H. M. "Diagnosis of Pneumonia from Chest X-Ray Images Using Deep Learning." EBBT, 2019.
2. Kundu, R., et al. "Pneumonia detection in chest X-ray images using an ensemble of deep learning models." PLoS ONE, 2021.


