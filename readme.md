# Flower Classification with CNN

This project implements a Convolutional Neural Network (CNN) to classify flower images into 5 categories. The model was developed in Google Colab using TensorFlow/Keras.

## Dataset
- **Source**: Custom flower dataset from Google Drive
- **Classes**: 5 flower types (specific names not shown in notebook)
- **Images**:
  - Training: 3,457 images (80% of total)
  - Validation: 860 images (20% of total)
- **Preprocessing**:
  - Resized to 150x150 pixels
  - Normalized pixel values (0-1 range)
  - Data augmentation (horizontal flips, 30° rotations, 20% zoom)

## Model Architecture
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

Total Parameters: 4,828,997

Optimizer: Adam

Loss Function: Categorical Crossentropy

Training
Epochs: 10

Batch Size: 32

Training Time: ~35 minutes (on Colab GPU)

Epoch	Train Accuracy	Val Accuracy	Train Loss	Val Loss
1	29.89%	51.05%	1.5647	1.1632
5	66.29%	66.40%	0.8872	0.8813
10	73.22%	69.77%	0.6818	0.8069
Results
Final Training Accuracy: 73.22%

Final Validation Accuracy: 69.77%

Confusion Matrix:
Confusion Matrix (example placeholder)

Visualization
Accuracy/Loss curves during training

Confusion matrix

Classification report (precision, recall, f1-score)

Usage
Open the Google Colab Notebook

Mount Google Drive with dataset at:
/content/drive/MyDrive/Dataset for Colab/flowers

Run all cells sequentially

Requirements
Python 3.x

TensorFlow 2.x

Keras

NumPy

Matplotlib

Seaborn

Future Improvements
Implement transfer learning (VGG16, ResNet)

Add more augmentation techniques

Hyperparameter tuning

Class imbalance handling

Deeper network architecture

Early stopping callbacks

License
MIT
