# Tamang ASL ✌️

A real-time American Sign Language (ASL) practice platform built with Flet and Python. Powered by a MobileNetV2 model fine-tuned via transfer learning, the application uses TensorFlow Lite to provide live camera feedback, interactive quizzes, and on-device gesture classification.

## 🌟 Features

* **Real-Time Classification:** High-performance local inference using TFLite and MediaPipe ensures zero-latency feedback without needing an internet connection.
* **Multiple Learning Modes:** Features a guided Practice mode, a randomized Quiz mode, and a Word Builder for comprehensive ASL practice.
* **Privacy First:** All webcam processing and neural network predictions are handled completely offline on your local machine.

## 📋 Prerequisites

* **Python 3.13** (Recommended)
* A working webcam

## 🚀 How to Use

**1. Clone the repository**

```bash
git clone https://github.com/Lejaaand/TamangASL.git
cd Tamang-ASL
```

**2. Create and activate a virtual environment**
* Windows
```bash
python -m venv venv
venv\Scripts\activate
```
* macOS
```bash
python3.13 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the application**
```bash
python TamangASL_Main.py
```
