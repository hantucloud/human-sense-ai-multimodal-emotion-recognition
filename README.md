# 🧠 HumanSense AI – Multimodal Emotion Recognition System

HumanSense AI is a **multimodal deep learning system** that analyzes  **human emotions using both facial expressions and audio signals** .

It integrates:

* 🎭 **Face Emotion Recognition (FER)** using FER-2013
* 🎤 **Speech Emotion Recognition (SER)** using RAVDESS
* 🔗 **Fusion Engine** for final emotion prediction

---

## 🚀 Features

* 🎥 Real-time face emotion detection
* 🎤 Audio-based emotion classification
* 🧠 Multimodal fusion for improved accuracy
* 📊 Confidence-based scoring system
* ⚡ Interactive Streamlit dashboard

---

## 🧩 System Architecture

The system consists of three major components:

* 🎭 Face Emotion Pipeline (FER-2013)
* 🎤 Audio Emotion Pipeline (RAVDESS)
* 🔗 Multimodal Fusion Engine

📁 Detailed architecture diagrams are available in the `docs/` folder:

* `FER-2013_Face_Pipeline.png`
* `Ravdess_Audio_Pipeline.png`
* `MultiModal_Fusion.png`

---

## 📂 Project Structure

```bash
HumanSense-AI/
├── app/                  # Streamlit application
├── docs/                 # Architecture diagrams
├── models/               # External model files (download required)
├── notebooks/            # Training & experimentation
├── results/              # Evaluation metrics & outputs
├── requirements.txt
└── README.md
```

---

## 📦 Model Files

⚠️ Due to size limitations, trained models are hosted externally.

👉 **Download Models:**
[https://drive.google.com/drive/folders/1SKODIOYHia3v4NBa8wBIywJeWsbampTC?usp=sharing](https://drive.google.com/drive/folders/1SKODIOYHia3v4NBa8wBIywJeWsbampTC?usp=sharing)

### Required Files:

* `audio_emotion_model.keras`
* `face_emotion_model.pth`
* `blaze_face_short_range.tflite`
* `face_landmarker.task`

📌 Place all files inside:

```
models/
```

---

## 📚 Research Foundation

### 🎤 Audio Dataset — RAVDESS

* Ryerson Audio-Visual Database of Emotional Speech and Song
* Contains **7356 recordings** from 24 professional actors
* Emotions: calm, happy, sad, angry, fearful, surprise, disgust
* Multi-modal: audio-only, video-only, and audio-visual formats

🔗 Official Dataset (Zenodo):
[https://zenodo.org/records/1188976](https://zenodo.org/records/1188976)

---

### 🎭 Face Dataset — FER-2013

* Grayscale facial images of size **48×48 pixels**
* 7 emotion classes:
  * Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
* Training set:  **28,709 images** , Test set: **3,589 images**

🔗 Dataset (Kaggle):
[https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

---

## 📈 Results

### Audio Model (RAVDESS)(94.44%)

* Strong fold accuracy across training
* Clear class separation in confusion matrix

### Face Model (FER-2013)(69.13%)

* Stable training performance
* Reliable classification on test data

📁 Refer to `/results` folder for:

* Confusion matrices
* Accuracy plots
* Training curves

---

## ⚡ Installation

```bash
git clone https://github.com/sandip234-ui/HumanSense-AI.git
cd HumanSense-AI
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app/app.py
```

---

## ⚠️ Important Note

Before running the application, ensure all required model files are downloaded and placed inside the `models/` directory.

---

## 🔮 Future Improvements

* 🔔 Real-time alert system
* ☁️ Cloud deployment
* 📱 Mobile integration
* 🧠 Advanced fusion strategies (attention-based / weighted fusion)

---

## 👨‍💻 Author

**Sandip Biswal**

* GitHub: [https://github.com/sandip234-ui](https://github.com/sandip234-ui)
* LinkedIn: [https://www.linkedin.com/in/sandip-biswal-728a7a291/](https://www.linkedin.com/in/sandip-biswal-728a7a291/)

---

## 📜 License

For educational and research purposes only.
