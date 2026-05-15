import io
import base64
import numpy as np
from PIL import Image, ImageDraw
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# ── Load model ─────────────────────────────────────────────────────────────────
model = YOLO("best.pt")

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="Earthquake Damage Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status" : "running",
        "model"  : "YOLOv11m-seg",
        "classes": ["collapsed", "intact"],
    }

# ── NMS for merging duplicate detections across patches ────────────────────────
def nms_detections(detections, iou_threshold=0.4):
    if not detections:
        return detections

    boxes  = np.array([d["bbox"] for d in detections], dtype=np.float32)
    scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        union = areas[i] + areas[order[1:]] - inter
        iou   = inter / (union + 1e-6)
        order = order[1:][iou < iou_threshold]

    return [detections[i] for i in keep]

# ── Sliding window inference ───────────────────────────────────────────────────
def sliding_window_predict(image: Image.Image, conf: float = 0.3,
                           patch_size: int = 320, overlap: int = 80):
    img_w, img_h   = image.size
    stride         = patch_size - overlap
    all_detections = []

    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x2 = min(x + patch_size, img_w)
            y2 = min(y + patch_size, img_h)

            patch   = image.crop((x, y, x2, y2))
            results = model.predict(patch, conf=conf, iou=0.5, verbose=False)
            result  = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                    # Translate patch coords to full image coords
                    bx1 += x; by1 += y
                    bx2 += x; by2 += y

                    # Clip to image bounds
                    bx1 = max(0, min(bx1, img_w))
                    by1 = max(0, min(by1, img_h))
                    bx2 = max(0, min(bx2, img_w))
                    by2 = max(0, min(by2, img_h))

                    all_detections.append({
                        "class"     : result.names[int(box.cls)],
                        "confidence": round(float(box.conf), 3),
                        "bbox"      : [round(bx1,2), round(by1,2),
                                       round(bx2,2), round(by2,2)],
                    })

            if x2 == img_w:
                break
            x += stride

        if y2 == img_h:
            break
        y += stride

    return nms_detections(all_detections)

# ── Draw detections on full image ──────────────────────────────────────────────
def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    draw   = ImageDraw.Draw(image, "RGBA")
    colors = {
        "collapsed": (230, 57,  70,  110),
        "intact"   : (42,  157, 143, 110),
    }
    border = {
        "collapsed": (230, 57,  70),
        "intact"   : (42,  157, 143),
    }

    for det in detections:
        cls  = det["class"]
        bbox = det["bbox"]
        conf = det["confidence"]
        draw.rectangle(bbox,
                       fill    = colors.get(cls, (200,200,200,100)),
                       outline = border.get(cls, (200,200,200)),
                       width   = 3)
        draw.text((bbox[0] + 4, bbox[1] + 4),
                  f"{cls} {conf:.2f}",
                  fill=border.get(cls, (255,255,255)))

    return image

# ── Predict endpoint ───────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents     = await file.read()
    image        = Image.open(io.BytesIO(contents)).convert("RGB")
    img_w, img_h = image.size

    # Sliding window for large images, direct for small
    if img_w > 640 or img_h > 640:
        detections = sliding_window_predict(image, conf=0.25,
                                            patch_size=640, overlap=100)
        annotated  = draw_detections(image.copy(), detections)
        mode       = "sliding_window"
    else:
        results    = model.predict(image, conf=0.25, iou=0.5, verbose=False)
        result     = results[0]
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                detections.append({
                    "class"     : result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "bbox"      : [round(v,2) for v in box.xyxy[0].tolist()],
                })
        annotated = Image.fromarray(result.plot())
        mode      = "direct"

    # Encode annotated image — resize if too large for browser
    if img_w > 2000 or img_h > 2000:
        scale     = 2000 / max(img_w, img_h)
        new_w     = int(img_w * scale)
        new_h     = int(img_h * scale)
        annotated = annotated.resize((new_w, new_h), Image.LANCZOS)

    buffer  = io.BytesIO()
    annotated.save(buffer, format="JPEG", quality=82)
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    counts = {"collapsed": 0, "intact": 0}
    for d in detections:
        counts[d["class"]] += 1

    return JSONResponse({
        "detections"      : detections,
        "total"           : len(detections),
        "counts"          : counts,
        "image_size"      : {"width": img_w, "height": img_h},
        "mode"            : mode,
        "annotated_image" : img_b64,
    })