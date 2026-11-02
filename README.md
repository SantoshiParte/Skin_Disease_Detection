# SkinSense AI – Skin Disease Detection

An AI-powered skin disease classification web application built using **Python, YOLO, and Streamlit**. The application allows users to upload a skin image and receive an AI-generated prediction with a confidence score and educational information about the predicted condition.

## 🌟 Project Overview

SkinSense AI is designed to demonstrate how deep learning and computer vision can assist in skin image classification.

The application uses a trained YOLO classification model to analyze uploaded images and display the predicted skin condition along with relevant information, possible causes, and precautions.

**Note:** This project is developed for educational and research purposes. It is not a substitute for professional medical diagnosis.

## ✨ Features

- **AI-Based Skin Classification:** Predicts the skin condition using a trained YOLO classification model.
- **Image Upload:** Upload skin images in JPG, JPEG, or PNG format.
- **Prediction Confidence:** Displays the model's prediction confidence.
- **Top Predictions:** Shows alternative predictions, where supported by the model.
- **Disease Information:** Provides educational information about conditions, possible causes, and precautions.
- **Prediction History:** View previous predictions during the application session.
- **Interactive Dashboard:** Clean and user-friendly Streamlit interface.
- **Downloadable Results:** Download prediction results for future reference.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLO | Image classification model |
| Ultralytics | YOLO model implementation |
| Streamlit | Web application interface |
| OpenCV | Image processing |
| Pillow | Image handling |
| NumPy | Numerical operations |

## 📁 Project Structure

```text
Skin_Disease_Detection/
│
├── app.py
│
├── models/
│   └── best.pt
│
├── utils/
│   ├── __init__.py
│   ├── predictions.py
│   └── disease_info.py
│
├── requirements.txt
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/SantoshiParte/Skin_Disease_Detection.git
```

### 2. Navigate to the Project Directory

```bash
cd Skin_Disease_Detection
```

### 3. Create a Virtual Environment

```bash
python -m venv skin_ai
```

Activate the environment:

**Windows:**
```bash
skin_ai\Scripts\activate
```

**macOS / Linux:**
```bash
source skin_ai/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the Trained Model

Place your trained YOLO classification model inside the `models` directory.

The expected model path is:

```text
models/best.pt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🚀 How to Use

1. Open the SkinSense AI web application.
2. Navigate to the Skin Analysis section.
3. Upload a skin image in JPG, JPEG, or PNG format.
4. Click the prediction button.
5. Wait for the AI model to analyze the uploaded image.
6. View the predicted condition and confidence score.
7. Explore the available disease information and precautions.
8. Download the prediction report if required.

## 🔮 Future Enhancements

- Improve classification performance using larger and more diverse datasets.
- Add explainable AI visualizations such as
