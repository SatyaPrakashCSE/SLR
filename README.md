# SLR – Sign Language Recognition

**Bridging Communication Gaps with Instant Sign Language**

SLR is an open-source **Sign Language Recognition (SLR)** project focused on real-time hand gesture recognition using **Computer Vision and Deep Learning**. The system enables sign language detection through a webcam by training and deploying a CNN-based model on 48×48 grayscale images.

---

## 📌 Overview

This project provides an end-to-end pipeline for building a real-time sign language recognition system, including:

- Dataset collection using a webcam  
- Image preprocessing and dataset splitting  
- CNN model training and evaluation  
- Real-time gesture prediction using OpenCV  

The goal is to assist communication for **deaf and hard-of-hearing individuals** by translating sign gestures into readable output.

---

## 🚀 Features

- 🧠 **CNN-Based Model** trained on 48×48 grayscale images  
- 📸 **Webcam Data Collection** for custom sign datasets  
- 📊 **Dataset Management** (train/validation split)  
- ⚡ **Real-Time Sign Detection** using OpenCV  
- 💾 **Pretrained Model Included** (`.json` + `.h5`)  

---

## 📂 Project Structure

```
SLR/
├── SignImage48x48/                       # Collected sign images (48x48)
├── splitdataset48x48/                   # Train/validation dataset
├── templates/                           # UI templates (if applicable)
├── app.py                               # Application entry point
├── collectdata.py                       # Webcam data collection script
├── realtimedetection.py                 # Live sign recognition
├── split.py                             # Dataset splitting script
├── trainmodel.ipynb                     # Model training notebook
├── signlanguagedetectionmodel48x48.json # Model architecture
├── signlanguagedetectionmodel48x48.h5   # Trained model weights
├── requirements.txt                     # Project dependencies
└── README.md                            # Project documentation
```

---

## 🛠️ Prerequisites

- Python 3.8 or higher  
- Pip package manager  
- Webcam (for data collection & real-time detection)

---

## ⚙️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/SatyaPrakashCSE/SLR
```

2. **Navigate to the project directory**
```bash
cd SLR
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📸 Data Collection

Use the script below to collect sign language images via webcam:

```bash
python collectdata.py
```

Images will be stored in the `SignImage48x48/` directory.

---

## 📊 Dataset Preparation

Split the dataset into training and validation sets:

```bash
python split.py
```

The output will be saved in `splitdataset48x48/`.

---

## 🧠 Model Training

Open the training notebook:

```bash
jupyter notebook trainmodel.ipynb
```

The notebook covers:
- Data preprocessing  
- CNN architecture  
- Model training & evaluation  
- Model export (`.json` and `.h5`)

---

## 🎥 Real-Time Detection

Run the real-time sign recognition system:

```bash
python realtimedetection.py
```

The webcam feed will display live predictions for detected hand signs.

---

## 📱 Application Mode (Optional)

If applicable, start the main application:

```bash
python app.py
```

---

## 🧪 Technologies Used

- Python  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- Jupyter Notebook  

---

## 📌 Notes

- Input images are standardized to **48×48 grayscale**
- Model files are included for direct inference
- Can be extended to support more gestures or words

---

## 📄 License

This project is open-source. Add a license file (MIT recommended) if you plan to distribute or reuse.

---

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome.  
Feel free to fork the repository and submit a pull request.
