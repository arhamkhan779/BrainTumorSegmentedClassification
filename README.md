# 🧠 BrainTumorSegmentedClassification

A deep learning-based system for **Brain Tumor Segmentation and Hierarchical Classification** leveraging **U-Net** for segmentation and **CNN** for classification. The project improves tumor diagnosis accuracy by focusing on the segmented tumor region and integrates **Explainable AI (Grad-CAM)** to enhance model transparency.

---

## 🚀 Project Motivation

Traditional classification models analyze entire MRI scans, which may include irrelevant regions. This project enhances prediction accuracy by:
- **Extracting tumor regions first (via segmentation)**
- **Classifying only the segmented tumor area**
- **Using Explainable AI to justify predictions**

---

## 📌 Project Strategy

### 🔹 Step 1 – Tumor Segmentation (U-Net)
- Train U-Net on labeled MRI segmentation dataset.
- Generate a binary tumor mask.

### 🔹 Step 2 – Hierarchical Tumor Classification (CNN)
- Apply mask to original MRI.
- Feed segmented region into CNN for classification.
- Predict tumor type and probabilities.

### 🔹 Step 3 – Fusion Architecture

```

MRI Image
│
▼
U-Net
(Segmentation Model)
│
▼
Segmented Tumor Region
│
▼
CNN / ResNet
(Classification Model)
│
▼
Final Prediction
(Glioma / Meningioma / Pituitary + Probabilities)

````

---

## 🧠 Hierarchical Classification Logic

| Level | Type of Prediction |
|-------|--------------------|
| 1️⃣ | Tumor vs No Tumor |
| 2️⃣ | Benign vs Malignant |
| 3️⃣ | Glioma / Meningioma / Pituitary (or other types) |

---

## 📂 Datasets

| Purpose | Dataset | Source |
|--------|---------|--------|
| Segmentation | Brain Tumor Segmentation Dataset | https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation |
| Classification | Brain Tumor MRI Dataset | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |

---

## 🧪 Model Architecture Summary

| Stage | Model | Output |
|-------|--------|--------|
| Segmentation | U-Net / U-Net++ | Tumor mask |
| Classification | Custom CNN / ResNet | Tumor class |
| Explainable AI | Grad-CAM / Grad-CAM++ | Visual heatmap |

---

## ⚙️ Tech Stack

```bash
Python
TensorFlow / PyTorch
OpenCV
U-Net, ResNet/EfficientNet
Grad-CAM (Explainable AI)
Flask (Optional Web Interface)
````

---

## 📦 Installation

```bash
git clone https://github.com/username/BrainTumorSegmentedClassification.git
cd BrainTumorSegmentedClassification
pip install -r requirements.txt
```

If using PyTorch:

```bash
pip install torch torchvision torchaudio
```

If using TensorFlow:

```bash
pip install tensorflow
```

---

## 🖥️ Usage

```python
from segmentation import UNetModel
from classification import CNNClassifier
from utils import apply_mask

# Load models
segmentation_model = UNetModel.load("unet_model.h5")
classification_model = CNNClassifier.load("cnn_model.h5")

# Process MRI image
mask = segmentation_model.predict(image)
segmented_image = apply_mask(image, mask)
prediction = classification_model.predict(segmented_image)
```

---

## 🌟 Explainable AI (Grad-CAM Example)

```python
from xai.grad_cam import GradCAM

gradcam = GradCAM(model=classification_model)
heatmap = gradcam.generate(segmented_image)

# Overlay heatmap
visualize_heatmap(segmented_image, heatmap)
```

---

## 📊 Expected Outcomes

| Metric                    | Goal                 |
| ------------------------- | -------------------- |
| Segmentation (Dice Score) | > 0.85               |
| Classification Accuracy   | > 90%                |
| Grad-CAM Trust Score      | Must highlight tumor |

---

## 📈 Folder Structure

```
BrainTumorSegmentedClassification/
│── data/
│   ├── segmentation_dataset/
│   └── classification_dataset/
│── models/
│   ├── unet.py
│   ├── cnn_classifier.py
│── xai/
│   └── grad_cam.py
│── utils/
│── notebooks/
│── app.py (optional Flask deployment)
│── README.md
│── requirements.txt
```

---

## 🔬 Future Enhancements

* Deploy as **Web-Based Medical Assistant**
* Integrate with **DICOM support**
* Use **Attention U-Net or YOLOv8-seg** for experimentation
* Automated PDF Report Generation (medical format)

---

## 📢 LinkedIn Post Suggestion

> 🚀 New AI Project: Brain Tumor Segmentation & Classification with Explainable Deep Learning
> Using **U-Net for segmentation** and **CNN for guided classification**, I developed a medical AI pipeline that improves tumor diagnosis accuracy. The system uses **Grad-CAM for explainability**, providing visual insights for decision-making.
> 🔧 Tech: TensorFlow, OpenCV, UNet, ResNet, Grad-CAM
> 📊 Results coming soon! | GitHub link below
> #AI #DeepLearning #ComputerVision #MedicalAI #ExplainableAI

---

## 🏁 Conclusion

This project demonstrates a **research-grade AI solution** for brain tumor diagnosis by using **segmentation-driven classification and explainable intelligence**. It aims to support early detection and improve clinical trust in AI.

---

## 👤 Author

**Arham Khan**
AI Engineer | Deep Learning Specialist
📍 Pakistan
🔗 GitHub: [https://github.com/arhamkhan779](https://github.com/arhamkhan779)
🔗 LinkedIn: [https://linkedin.com/in/arhamkhannn](https://linkedin.com/in/arhamkhannn)
