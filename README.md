# WildLife-Detection

A real-time, long-range wildlife detection and notification system that captures motion-triggered images on remote ESP32-CAM nodes, transmits them over Wi-Fi HaLow to a Raspberry Pi base station for on-device YOLOv8 inference (Hailo-8L accelerator), verifies and labels species using Anthropic's Claude vision API, uploads results to Firebase, and visualizes everything in a React dashboard.

---

## Overview

This project combines low-power embedded hardware, edge AI, cloud storage, and a modern web dashboard to monitor wildlife activity in remote outdoor areas where standard Wi-Fi cannot reach.

```
┌──────────────┐   HaLow Wi-Fi    ┌────────────────┐   Cloud   ┌──────────────┐
│ ESP32-CAM(s) │ ───────────────► │ Raspberry Pi   │ ────────► │  Firebase    │
│ + PIR sensor │   (long-range)   │ + Hailo-8L NPU │           │   Storage    │
└──────────────┘                  └────────────────┘           └──────────────┘
                                          │                            │
                                          │ Claude API                 │
                                          ▼                            ▼
                                   Species ID + SMS         React Dashboard
```

---

## Features

- **Motion-activated capture** — PIR-triggered ESP32-CAM nodes wake from idle, capture a 3-frame burst, and stream JPEGs only when activity is detected
- **Long-range wireless** — Wi-Fi HaLow (802.11ah) link reaches hundreds of meters, ideal for forests, fields, and remote properties
- **Edge AI inference** — YOLOv8s running on a Hailo-8L NPU on the Raspberry Pi for fast, low-power detection
- **Wildlife class filter** — only triggers on relevant COCO classes (people, birds, mammals); ignores cars, furniture, etc.
- **Claude-powered species ID** — sends frames containing detections to Claude for fine-grained species identification (e.g. *white-tailed deer* instead of just *deer*) with confidence score
- **Cloud storage** — annotated frames automatically uploaded to Firebase Storage, organized by camera and motion event
- **Email/SMS alerts** — sends instant notifications via Gmail SMTP or carrier email-to-SMS gateway, with per-camera cooldown to prevent spam
- **React dashboard** — browse detections by camera, animal, time of day, and date with charts and filters
- **Robust networking** — TCP keepalives, ACK/NAK protocol, session generations, heartbeats, and automatic reconnect with exponential backoff

---

## Hardware

| Component | Purpose |
|---|---|
| ESP32-CAM (Heltec HaLow variant) | Camera node — captures frames and transmits over HaLow |
| PIR motion sensor | Wake trigger for the camera node |
| Raspberry Pi (4 or 5 recommended) | Base station running the listener and inference pipeline |
| Hailo-8L AI accelerator (M.2 / HAT) | Hardware NPU for YOLOv8 inference |
| HaLow access point | Long-range Wi-Fi backbone connecting nodes to the Pi |

---

## Repository Layout

```
WildLife-Detection/
├── ESP32_SENDER_CODE/
│   └── HalowClient.ino           # ESP32-CAM firmware (PIR-triggered capture + TCP upload)
│
├── RPI5_LISTENER_CODE/
│   ├── halow_listener.py         # Raspberry Pi server: receives frames, runs YOLO, calls Claude
│   ├── hailo_utils.py            # HailoYOLO wrapper for Hailo-8L NPU inference
│   └── yolov8s.hef               # Compiled YOLOv8s model for Hailo NPU
│
├── YOLOv8s/
│   └── coco.txt                  # COCO class label list
│
└── wildlife-dashboard/
    ├── App.jsx                   # React dashboard for browsing detections
    ├── App.css                   # Global styles
    ├── firebase.js               # Firebase Storage configuration
    ├── main.jsx                  # React entry point
    └── index.css
```

---

## Setup

### 1. Raspberry Pi base station

Install dependencies:

```bash
sudo apt update
sudo apt install python3-pip python3-opencv
pip install hailo-platform firebase-admin anthropic numpy
```

You will also need the Hailo runtime drivers and HailoRT installed for the Hailo-8L accelerator. Follow [Hailo's official setup guide](https://hailo.ai/developer-zone/) for your Pi.

Create the output directory and copy the model:

```bash
mkdir -p /home/pi/Public/WildLife-Detection/Images
cp yolov8s.hef coco.txt /home/pi/Public/WildLife-Detection/YOLOv8s/
```

### 2. Configuration

Edit the constants at the top of `halow_listener.py`:

```python
ANTHROPIC_API_KEY = "your-claude-api-key"
FIREBASE_CERT     = "path/to/serviceAccount.json"
FIREBASE_BUCKET   = "your-project.firebasestorage.app"
GMAIL_ADDRESS     = "your-alert-sender@gmail.com"
GMAIL_APP_PASSWORD = "your-gmail-app-password"
ALERT_TO_ADDRESS  = "recipient@example.com"   # or carrier SMS gateway
SMS_ENABLED       = True
```

Run the listener:

```bash
python3 halow_listener.py
```

The server listens on port `8080` for incoming camera connections.

### 3. ESP32-CAM nodes

Open `HalowClient.ino` in the Arduino IDE and configure each node:

```cpp
const char*  CAM_ID   = "A";                      // unique per unit (A, B, C...)
const int    PIR_PIN  = 1;                        // PIR sensor GPIO
IPAddress    LOCAL_IP(192, 168, 100, 116);        // unique per unit

const char*  SSID     = "Wildlife_Halow";
const char*  PASSWORD = "your-halow-password";
const char*  HOST     = "192.168.100.1";          // Raspberry Pi HaLow IP
const int    PORT     = 8080;
```

Install the [HaLow Arduino library](https://github.com/HelTec-Aaron-Lee/HaLow) and the ESP32 board package. Flash the firmware to each ESP32-CAM unit.

### 4. React dashboard

```bash
cd wildlife-dashboard
npm install
npm run dev
```

Update `firebase.js` with your own Firebase project credentials. The dashboard reads images from the `detected/` and `empty/` folders in your storage bucket.

---

## How It Works

1. **Trigger** — The PIR sensor on an ESP32-CAM detects motion, waking the node from idle
2. **Capture** — After a brief settle delay, the node captures a burst of 3 JPEGs ~1.5 seconds apart
3. **Transmit** — Each frame is sent over Wi-Fi HaLow to the Raspberry Pi using a length-prefixed TCP protocol with ACK/NAK flow control
4. **Detect** — The Pi runs YOLOv8s on the Hailo-8L NPU, filtering for animal/person classes
5. **Identify** — If a relevant detection is found, the frame is sent to Claude's vision API for fine-grained species identification
6. **Save & upload** — Annotated frames are saved locally and uploaded to Firebase under `detected/` (with bounding boxes) or `empty/` (no detection)
7. **Notify** — An email alert is sent for each new detection, with a per-camera cooldown
8. **Visualize** — The React dashboard displays all detections with filters for camera, species, time of day, and date

---

## Detection Classes

The system filters YOLO detections to these COCO classes:

`person`, `bird`, `cat`, `dog`, `horse`, `sheep`, `cow`, `elephant`, `bear`, `zebra`, `giraffe`

Claude then provides finer-grained species identification beyond these coarse categories.

---

## Security Note

The configuration constants in `halow_listener.py` and `firebase.js` are placeholders. Before deploying:

- Move secrets (Claude API key, Gmail app password, Firebase credentials) to environment variables or a `.env` file
- Add `.env`, `serviceAccount.json`, and any keys to `.gitignore`
- Rotate any keys that have been committed to git history

---

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection model
- [Hailo](https://hailo.ai) for the edge AI accelerator and runtime
- [Anthropic Claude](https://www.anthropic.com) for vision-based species identification
- [Heltec HaLow](https://heltec.org) for the long-range Wi-Fi modules
