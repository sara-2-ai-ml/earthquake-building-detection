<div align="center">

# 🛰️ Earthquake Building Damage Detection
### AI-Powered Satellite Imagery Analysis · 2023 Türkiye Earthquake

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Instance_Segmentation-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white)](https://ultralytics.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)

<br/>

> **When 346,000 buildings collapse in seconds, waiting for a manual survey is a death sentence.**
>
> The 2023 Türkiye earthquake killed over 50,000 people. In the hours after, emergency teams worked blind —
> damage assessment takes days, but survival windows close in hours.
>
> **This system was built to change that.**

<br/>

</div>

---

## 🖼️ Screenshots

### 🌍 Satellite Imagery — Training Data
*Real Maxar satellite tiles used for model training (Hatay, Adıyaman, Kahramanmaraş)*

<div align="center">
<img src="frontend/public/data/10300500D9F8D200-visual K 16.png" width="48%" />
<img src="frontend/public/data/10300500D9F8D200-visual K 18.png" width="48%" />
</div>

<div align="center">
<img src="frontend/public/data/10300500D9F8D500-visual Hayat(2) 16.png" width="32%" />
<img src="frontend/public/data/10300500D9F8D500-visual Hayat(2) 17.png" width="32%" />
<img src="frontend/public/data/10300500D9F8D500-visual Hayat(2) 18.png" width="32%" />
</div>

---

### 💻 Web Application — Live Demo
*FastAPI + React interface for real-time damage detection*

<div align="center">
<img src="frontend/public/website/Screenshot 2026-05-15 005120.png" width="100%" />
<br/><br/>
<img src="frontend/public/website/Screenshot 2026-05-15 005149.png" width="49%" />
<img src="frontend/public/website/Screenshot 2026-05-15 005806.png" width="49%" />
</div>

---

## 📊 Performance at a Glance

<div align="center">

| Metric | Score |
|--------|-------|
| **Mask mAP50** | **0.808** |
| **Recall** | **0.753** |
| **Inference Speed** | **< 10ms / image** |
| **Cross-Validation (5-Fold)** | **0.800 ± 0.018** |
| **Annotated Instances** | **2,500+** |
| **Training Epochs** | **150** |

</div>

---

## 🎯 The Problem

The 2023 Türkiye–Syria earthquake was one of the deadliest natural disasters of the 21st century:

- **50,000+ lives lost**
- **346,000 buildings** collapsed or severely damaged
- **Emergency teams working blind** — traditional damage assessment requires days of ground surveys
- **The critical window**: 72 hours after collapse, survival probability drops dramatically

**Manual satellite image analysis is too slow.** A human analyst reviewing thousands of satellite tiles cannot produce actionable damage maps fast enough to guide rescue teams in real time.

---

## 💡 The Solution

An end-to-end AI system that:

1. **Ingests** high-resolution Maxar satellite imagery
2. **Detects** collapsed vs. intact buildings using instance segmentation
3. **Generates** damage maps with bounding polygons in under 10ms per image
4. **Deploys** as a web application accessible to emergency response teams

```
Satellite Image (Maxar)
        ↓
  React Frontend
  ─ Upload interface
  ─ Interactive damage map
        ↓
  FastAPI Backend
  ─ Sliding window preprocessing
  ─ YOLO inference pipeline
        ↓
  YOLOv11m-seg Model
  ─ Collapsed / Not Collapsed
  ─ Instance masks + confidence scores
        ↓
  Real-time Damage Assessment
```

---

## 🗂️ Project Structure

```
EARTHQUAKE-ML/
├── backend/
│   ├── main.py                  # FastAPI server + inference pipeline
│   └── requirements.txt         # Python dependencies
│
└── frontend/
    ├── src/
    │   └── components/          # React components
    ├── public/
    │   ├── data/                # Sample satellite imagery
    │   └── website/             # App screenshots
    ├── index.html
    ├── vite.config.js
    └── package.json
```

> ⚠️ **Model file** (`best.pt`) is hosted separately due to size — see download instructions below.

---

## 🏗️ Architecture & Methodology

### 1. Data Collection & Annotation

**No pre-labeled dataset was used.** Every annotation was created manually.

- **Source**: Maxar satellite imagery (post-earthquake, Feb 2023)
- **Regions**: Hatay (Antakya), Adıyaman, Kahramanmaraş
- **Annotation tool**: Roboflow with Smart Polygon (SAM-powered)
- **Total instances**: 2,500+ manually annotated building polygons
- **Classes**: `collapsed` | `not_collapsed`
- **Format**: Instance segmentation (polygon masks, not bounding boxes)

### 2. Model Architecture

| Component | Choice | Reason |
|-----------|--------|--------|
| **Base Model** | YOLOv11m-seg | Best speed/accuracy tradeoff for satellite imagery |
| **Task** | Instance Segmentation | Precise building outlines, not just boxes |
| **Input size** | 640×640 | Standard for aerial imagery |
| **Inference** | Sliding window | Handles large satellite tiles without detail loss |

### 3. Training Configuration

```python
model = YOLO("yolov11m-seg.pt")
model.train(
    data="data.yaml",
    epochs=150,
    imgsz=640,
    cls=4.0,          # Higher weight for collapsed class (imbalanced data)
    dropout=0.1,      # Regularization for generalization
    copy_paste=0.6,   # Augmentation for sparse collapsed instances
    fl_gamma=2.0,     # Focal loss for class imbalance
)
```

### 4. Validation Strategy

To ensure statistical reliability on a limited dataset:

- **K-Fold Cross-Validation (k=5)** across all 2,500 annotated instances
- Result: **Mask mAP50 = 0.800 ± 0.018** (low variance = reliable model)
- Final model selected from best fold checkpoint

### 5. Data Augmentation

Critical for satellite imagery where orientation is arbitrary:

| Augmentation | Value | Reason |
|---|---|---|
| Rotation | 360° | Satellites image from any azimuth |
| Horizontal/Vertical Flip | ✅ | No canonical orientation |
| Mosaic | ✅ | Improves small object detection |
| Brightness | ±20% | Varying illumination conditions |
| Hue/Saturation | ✅ | Seasonal and atmospheric variation |
| Cutout | ✅ | Robustness to partial occlusion |

---

## 📈 Results

### Training Curves

| Loss | Behavior |
|------|----------|
| **Box Loss** | 1.9 → 0.8 (good convergence) |
| **Seg Loss** | 4.3 → 1.4 (model learned mask shapes) |
| **Class Loss** | 12.0 → 2.0 (strong class differentiation) |

### Final Metrics

```
Mask mAP50:      0.808  ✅
Mask mAP50:95:   0.481  ✅
Box mAP50:       0.821  ✅
Recall:          0.753  ✅  ← Priority metric for disaster response
Precision:       0.812  ✅
Inference:       <10ms  ✅  per 640×640 tile
```

> **Why Recall is the priority metric**: In disaster response, a False Negative (missing a collapsed building) 
> costs lives. A False Positive (flagging an intact building) wastes resources. 
> The model is tuned to maximize Recall while maintaining acceptable Precision.

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
Node.js 18+
```

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

**Download the trained model:**

📥 [Download best.pt from Google Drive](https://drive.google.com/drive/u/0/folders/1MaB0CRII8l8Bl_GsgGCzGH3GUtyoQPp6)

Place `best.pt` inside the `backend/` folder, then:

```bash
uvicorn main:app --reload
# Server running at http://localhost:8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# App running at http://localhost:5173
```

### API Usage

```python
import requests

# Upload a satellite image for analysis
with open("satellite_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )

result = response.json()
print(f"Collapsed buildings detected: {result['collapsed_count']}")
print(f"Confidence scores: {result['confidences']}")
```

---

## 🔧 Tech Stack

### Backend
| Tool | Purpose |
|------|---------|
| **FastAPI** | REST API server |
| **Ultralytics YOLOv11** | Inference engine |
| **OpenCV** | Image preprocessing & sliding window |
| **Python 3.10** | Core language |

### Frontend
| Tool | Purpose |
|------|---------|
| **React + Vite** | UI framework |
| **Leaflet.js** | Interactive satellite map |
| **Axios** | API communication |

### Data & Training
| Tool | Purpose |
|------|---------|
| **Maxar** | Satellite imagery source |
| **Roboflow** | Annotation & dataset management |
| **Google Colab** | GPU training environment |
| **Jupyter Notebook** | Experiment tracking |

---

## 🌍 Real-World Impact & Future Applications

| Use Case | Description |
|----------|-------------|
| **🚑 Emergency Response** | Prioritize rescue teams to highest-damage zones within minutes of disaster |
| **🗺️ Damage Mapping** | Generate city-wide heatmaps of collapsed structures automatically |
| **📋 Insurance Assessment** | Evaluate claims at scale without field surveyors in dangerous zones |
| **🏗️ Reconstruction Monitoring** | Track rebuilding progress over months from satellite archives |
| **📐 Urban Planning** | Identify structural vulnerabilities for updated building codes |
| **🌐 Conflict Monitoring** | Adapt to assess war damage in inaccessible areas |

---

## ⚠️ Known Limitations

- **Domain gap**: Model trained on earthquake damage; performance may degrade on war damage (different visual patterns)
- **Small buildings**: Instance segmentation of very small structures (<10px) remains challenging
- **Cloud cover**: Images with heavy cloud cover require filtering before inference
- **Dataset size**: 2,500 instances is production-viable but larger datasets would improve generalization

---

## 📥 Model Download

The trained model (`best.pt`, ~45MB) is hosted on Google Drive due to GitHub file size limits.

**[⬇️ Download best.pt](https://drive.google.com/drive/u/0/folders/1MaB0CRII8l8Bl_GsgGCzGH3GUtyoQPp6)**

After downloading, place it in the `backend/` folder:
```
EARTHQUAKE-ML/
└── backend/
    ├── best.pt   ← here
    └── main.py
```

---

## 📓 Training Notebook

The full training pipeline — data loading, augmentation, YOLOv11 training, K-Fold validation, metrics, and result visualization — is available on Google Colab:

**[🚀 Open in Google Colab](https://colab.research.google.com/drive/1g0d1PlEll3ZSmcPJ5ecryHx2_RFm-tZM?authuser=1#scrollTo=bThLfHF6P1F0)**

Includes:
- 📊 Training curves (loss, mAP, Recall)
- 🗂️ Dataset visualization with annotated masks
- 🔁 5-Fold cross-validation results
- 🧪 Inference on test images

---

## 📁 Dataset

The dataset was built from scratch using:
- **Maxar Open Data Program** (post-disaster satellite imagery, publicly available)
- **Manual polygon annotation** (2,500+ instances)
- **Locations**: Antakya/Hatay, Adıyaman, Kahramanmaraş — epicenter regions of the Feb 6, 2023 earthquake

Dataset available on Roboflow Universe: *[link coming soon]*

---

## 📚 References

- [Maxar Open Data Program](https://www.maxar.com/open-data)
- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com)
- [xBD: A Dataset for Assessing Building Damage from Satellite Imagery](https://xview2.org/)
- [Roboflow Universe — Aerial Building Detection](https://universe.roboflow.com)

---

## 👩‍💻 Author

**Sara Resulaj** — AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com)

---

<div align="center">

*Built with the conviction that AI should save lives, not just optimize metrics.*

**⭐ Star this repo if you believe technology can make disaster response faster and smarter.**

</div>
