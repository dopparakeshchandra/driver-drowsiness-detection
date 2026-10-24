import os, csv, cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import defaultdict
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

VIDEO_DIR = "my_videos"
LABELS = "labels.csv"
USE_CLAHE = False     # set True later to test contrast enhancement
THRESH = 0.5
MARGIN = 0.5          # seconds ignored at each end of a part
STEP = 2              # use every 2nd frame to save time

model = tf.keras.models.load_model("cnn_simple.keras")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
LEFT = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT = [362, 263, 387, 386, 385, 373, 374, 380]

def parse_t(s):
    s = s.strip()
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)

def make_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)

def crop_eye(gray, lm, idx, w, h):
    xs = [lm[i].x * w for i in idx]
    ys = [lm[i].y * h for i in idx]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = (max(xs) - min(xs)) * 1.6
    x1, x2 = int(max(cx - side / 2, 0)), int(min(cx + side / 2, w))
    y1, y2 = int(max(cy - side / 2, 0)), int(min(cy + side / 2, h))
    eye = gray[y1:y2, x1:x2]
    if eye.size == 0:
        return None
    eye = cv2.resize(eye, (96, 96))
    return np.stack([eye] * 3, axis=-1).astype("float32")

segments = defaultdict(list)
with open(LABELS, newline="") as f:
    for r in csv.DictReader(f):
        segments[r["video"].strip()].append({
            "start": parse_t(r["start"]), "end": parse_t(r["end"]),
            "truth": 1 if r["state"].strip() == "closed" else 0,
            "lighting": r["lighting"].strip(), "glasses": r["glasses"].strip()})

rows = []
for video, segs in segments.items():
    path = os.path.join(VIDEO_DIR, video)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("cannot open:", path)
        continue
    landmarker = make_landmarker()
    i = -1
    last_ms = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % STEP:
            continue
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ms = int(t * 1000)
        if ms <= last_ms:
            ms = last_ms + 1
        last_ms = ms
        seg = next((s for s in segs
                    if s["start"] + MARGIN <= t <= s["end"] - MARGIN), None)
        if seg is None:
            continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        p = None
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            if USE_CLAHE:
                gray = clahe.apply(gray)
            eyes = [e for e in (crop_eye(gray, lm, LEFT, w, h),
                                crop_eye(gray, lm, RIGHT, w, h)) if e is not None]
            if eyes:
                p = float(model(np.array(eyes), training=False).numpy().mean())
        rows.append({"truth": seg["truth"], "p": p,
                     "lighting": seg["lighting"], "glasses": seg["glasses"]})
    cap.release()
    print("done:", video)

def summarize(items):
    scored = [r for r in items if r["p"] is not None]
    if not items:
        return "no frames"
    if not scored:
        return f"frames={len(items)}, no face found"
    y = np.array([r["truth"] for r in scored])
    pred = (np.array([r["p"] for r in scored]) > THRESH).astype(int)
    rec = (pred[y == 1] == 1).mean() if (y == 1).any() else float("nan")
    fa = (pred[y == 0] == 1).mean() if (y == 0).any() else float("nan")
    return (f"frames={len(items)}, face found={len(scored)/len(items):.0%}, "
            f"accuracy={(pred == y).mean():.1%}, closed-eye recall={rec:.1%}, "
            f"false alarm on open={fa:.1%}")

lines = [f"USE_CLAHE={USE_CLAHE}, THRESH={THRESH}", "",
         "OVERALL: " + summarize(rows), ""]
for key in ("lighting", "glasses"):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    for g, items in groups.items():
        lines.append(f"{key}={g}: " + summarize(items))
    lines.append("")
text = "\n".join(lines)
print("\n" + text)
with open("my_results.txt", "w") as f:
    f.write(text + "\n")