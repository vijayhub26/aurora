
# Aurora 🌟
Gesture-Based Smart Home Automation using Jetson Orin Nano

Aurora is a real-time hand gesture recognition system that enables control of smart home appliances using simple hand gestures. Designed to run on edge devices like the Jetson Orin Nano, this project uses deep learning and computer vision for intuitive, touchless home automation.

---

## 🔧 Tech Stack

- **Jetson Orin Nano** – Edge deployment and inference
- **TensorFlow** – MobileNetV3 model for gesture recognition
- **OpenCV** – Real-time image processing and gesture detection
- **Python** – Scripting and control logic
- **TensorRT** *(WIP)* – For performance optimization on Jetson

---

## 🎯 Features

- Detects **open/closed hand gestures** in real-time via webcam
- Controls devices like:
  - Lights
  - Television
  - Air Conditioner
- Designed for **low-latency** inference on edge devices
- Modular codebase for future integration with other appliances

---

## 📸 Demo

*(Coming Soon: Add a gif/video of the system in action)*  
> Example: Open hand turns on the light, closed hand turns it off.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/vijayhub26/aurora.git
cd aurora
```

### 2. Install Dependencies
Make sure Python 3.x is installed.

```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install opencv-python tensorflow numpy
```

### 3. Run the Gesture Recognition System
```bash
python3 gesture_detect.py
```

> Ensure your webcam is connected and the Jetson environment is properly configured.

---

## 🛠️ In Progress

- 🔄 Converting model to **TensorRT** for 2x faster inference
- 📱 Adding interface to mobile/voice controls
- 📊 Gesture recognition accuracy tuning

---

## 📚 Future Scope

- Integrate with **Google Assistant** or **Alexa**
- Expand gesture vocabulary (e.g., two-finger swipe)
- Add **multi-device mapping dashboard** via web interface

---

## 🤝 Contribution

Contributions, feedback, and suggestions are welcome!  
Feel free to fork the repo and raise a pull request.

---

## 📬 Contact

**Vijay Bharath V**  
📧 vijaybharath.vc@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/vijay-bharath26)

---

## ⭐️ Star this repo if you found it interesting!
