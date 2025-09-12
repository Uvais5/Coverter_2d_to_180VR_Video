# VR180 Video Conversion Project

This project provides a pipeline to convert standard videos into **VR180 SBS (Side-by-Side)** format for immersive VR experiences.  
It uses **Gradio** for a simple web interface and supports GPU acceleration (CUDA).

---
## video 
[![Watch the demo](demo-thumbnail.png)](https://github.com/Uvais5/Coverter_2d_to_180VR_Video/blob/master/VR_video.mp4)

---
## ✨ Features

**Upload 2D Video** – Users can provide any short MP4 video.  

**AI Depth Estimation** – Uses **MiDaS models (DPT_Large, DPT_Hybrid)** for monocular depth prediction.  

**Stereo Rendering** – Generates left/right images to simulate binocular vision.  

**VR180 Conversion** – Produces **Top-Bottom (TB)** or **Side-by-Side (SBS)** stereoscopic VR formats.  

**Audio Preservation** – Original audio is synced into the converted video.  

**Web UI** – Built with **Gradio** for a simple, no-code interface.  

**CUDA GPU Support** (if available) for faster processing.  

**Preview Option** – Check a short preview before downloading the full video.  

---

## 🛠 Tech Stack

- **Backend**: Python, FastAPI (via Gradio)  
- **AI Models**: MiDaS (Depth Estimation)  
- **Libraries**:  
  - `torch`, `torchvision` (Deep Learning)  
  - `opencv-python` (Video Processing)  
  - `moviepy` (Audio/Video Sync)  
  - `numpy`  
  - `gradio` (Web UI)  

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/vr180-converter.git
   cd vr180-converter
   ```
2. Create a virtual environment:
```
python -m venv appenv
source appenv/bin/activate   # On Linux/Mac
appenv\Scripts\activate      # On Windows
```
3. Install dependencies:

```
pip install -r requirements.txt
```
4. Run the app:
```
python app.py
```
