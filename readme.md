# 🌸 Flower Classification with CNN

This project implements a **Convolutional Neural Network (CNN)** to classify flower images into 5 categories. The model was developed in **Google Colab** using **TensorFlow/Keras**.

---

## 📁 Dataset

- **Source**: [Download a flower dataset stored on Google Drive  ](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
- **Classes**: 5 flower types *(specific names not shown in notebook)*  
- **Total Images**: ~4,317  
  - **Training Set**: 3,457 images (80%)  
  - **Validation Set**: 860 images (20%)  

### 🔄 Preprocessing
- Resized images to **150x150 pixels**
- Normalized pixel values to range **[0, 1]**
- Applied **data augmentation**:
  - Horizontal flip
  - Rotation (30°)
  - Zoom (20%)

---

## 🧠 Model Architecture

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

