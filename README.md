# 🚀 DetectWear: AI Based Human-Centric Wearable Number Detection In CCTV Streams

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Video_Processing-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8n_%7C_11n-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg)
![CustomTkinter](https://img.shields.io/badge/UI-CustomTkinter-darkblue.svg)

> An intelligent, AI-powered CCTV surveillance framework designed to automatically detect personnel, recognize wearable identification numbers (like jerseys or industrial uniforms), and generate real-time compliance alerts in secure environments.

This project was developed as a technical initiative during an internship at the **Satish Dhawan Space Centre (SDSC) SHAR, Indian Space Research Organization (ISRO)**.

---

## 📑 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Built With](#-built-with)
- [Getting Started](#-getting-started)
- [Usage](#-usage)


---

## 📖 About the Project

Existing CCTV surveillance systems rely heavily on manual monitoring, which is inefficient for tracking workforce compliance in large, secure industrial zones. Furthermore, traditional OCR-based number recognition systems often trigger false alarms by detecting irrelevant environmental numbers (e.g., machinery labels, signboards, and vehicles).

**DetectWear** introduces a human-centric, two-stage deep learning approach:
1. It first detects humans in the surveillance feed.
2. It then restricts number recognition *exclusively* to the detected human regions.

This context-aware filtering drastically reduces false positives, ensuring that only valid worker identification numbers are logged and verified.

---

## ✨ Key Features

- **🎯 Context-Aware Detection:** Combines human detection and number recognition to ignore background text and machinery labels.
- **🚨 Automated Compliance Alerts:** Generates real-time, on-screen warnings when a detected person is missing a visible identification number.
- **📸 Evidence Logging:** Automatically captures snapshots of non-compliant individuals and maintains a detailed CSV incident log.
- **🖥️ Interactive GUI:** A CustomTkinter-based dashboard for uploading videos, connecting live CCTV/RTSP streams, and viewing dual-feed (original vs. processed) video.
- **📊 Real-Time Analytics:** Displays live statistics including FPS, detected humans, recognized jerseys, and compliance alert counts.
- **💾 Output Management:** Supports exporting processed feeds as MP4 videos and downloading alert logs as CSV files.

---

## 🏗️ System Architecture

The pipeline processes video streams sequentially for high accuracy and efficiency:

1. **Video Stream Acquisition:** Ingests live CCTV/RTSP streams or pre-recorded video files.
2. **Human Localization:** Uses **YOLOv8n** to extract bounding boxes around human subjects.
3. **ROI Extraction:** Crops the human regions to isolate the worker.
4. **Number Recognition:** Analyzes the cropped region using **YOLO11n** to detect wearable numbers.
5. **Compliance Verification:** Checks for the presence of the number; if missing, triggers the Alert Generation Pipeline.
6. **Visualization:** Overlays bounding boxes, IDs, and alert status on the live UI.

<img width="351" height="622" alt="image" src="https://github.com/user-attachments/assets/5ccab168-edb5-4924-91e2-02fbb06cdde7" />


---

## 🛠️ Built With

- **[Python](https://www.python.org/):** Core system programming.
- **[OpenCV](https://opencv.org/):** Video processing and frame manipulation.
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics):** YOLOv8n (Human Detection) & YOLO11n (Wearable Number Detection).
- **[PyTorch](https://pytorch.org/):** GPU-accelerated model inference.
- **[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter):** Modern, interactive GUI development.

---

## 🚀 Getting Started

Follow these instructions to set up the project locally.

### Prerequisites
- Python 3.8 or higher
- CUDA-enabled GPU (Highly recommended for real-time inference)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/ParinitaMalisetty/DetectWear-AI.git](https://github.com/ParinitaMalisetty/DetectWear-AI.git)
   cd DetectWear-AI
   ```
2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
3. Install the required dependencies:
```bash
pip install -r requirements.txt
```
Ensure your trained models (yolov8n.pt and yolo11n.pt) are placed in the models/ directory as structured in the codebase.

---

### 💻 Usage
1. Launch the interactive GUI application:
```bash
python jersey_detection_ui.py
```
2. Upload Video or Connect CCTV: Paste an RTSP link or select a local.
3. Click Start Detection to initialize the AI pipeline.
4. Monitor the dual-feed displays, real-time analytics, and the No-Jersey Alerts panel.
5. Use the Export Alert Log CSV button to download compliance reports, and check the alerts/ folder for evidence images.

<img width="993" height="587" alt="image" src="https://github.com/user-attachments/assets/6a1d796f-4fd5-4937-b15f-844799caa595" />

<img width="984" height="579" alt="image" src="https://github.com/user-attachments/assets/26843e78-49fb-4096-af1d-334f892afe27" />


Designed and Developed by Parinita Malisetty.


