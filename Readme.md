# Ghost / Invisibility Mode

A real-time computer vision project built with Python, OpenCV, and MediaPipe.

**Developer:** tubakhxn

---

## How It Works

The system captures a clean background frame during calibration, then uses MediaPipe Selfie Segmentation to isolate the person from the scene. When the invisibility effect is triggered, the segmented person region is replaced with the stored background, making them disappear in real time.

Hand gesture detection runs in parallel using MediaPipe Hands. Spreading both hands apart generates a bounding box between the index fingers. Pinching the thumb and index finger together on either hand toggles the invisibility effect on and off.

---

## Requirements

- Python 3.8 or higher
- Webcam

Dependencies are installed automatically on first run. To install manually:

```
pip install opencv-python numpy mediapipe
```

---

## Usage

```
python main.py
```

To specify a camera index:

```
python main.py 1
```

---

## Controls

| Key | Action |
|-----|--------|
| R | Recalibrate background |
| S | Save screenshot |
| Q / ESC | Quit |

---

## Gesture Reference

1. Stand still for 3 seconds during calibration
2. Show both hands spread apart — a yellow bounding box appears between your index fingers
3. Pinch thumb tip and index tip together on either hand — you vanish
4. Pinch again to reappear

---

## Tech Stack

- Python
- OpenCV
- MediaPipe (Selfie Segmentation + Hands)
- NumPy