import os, csv, cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import defaultdict
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

VIDEO_DIR = "my_videos"
LABELS = "labels.csv"
MARGIN = 0.5
STEP = 2
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]

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

def predict(gray, lm, w, h):
    eyes = [e for e in (crop_eye(gray, lm, LEFT, w, h),
                        crop_eye(gray, lm, RIGHT, w, h)) if e is not None]
    if not eyes:
        return None
    return float(model(np.array(eyes), training=False).numpy().mean())

segments = defaultdict(list)
with open(LABELS, newline="") as f:
    for r in csv.DictReader(f):
        segments[r["video"].strip()].append({
            "start": parse_t(r["start"]), "end": parse_t(r["end"]),
            "truth": 1 if r["state"].strip() == "closed" else 0,
            "lighting": r["lighting"].strip(), "glasses": r["glasses"].strip()})

rows = []
for vid_idx, (video, segs) in enumerate(segments.items(), start=1):
    path = os.path.join(VIDEO_DIR, video)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("cannot open:", path)
        continue
    landmarker = make_landmarker()
    i, last_ms = -1, -1
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
        p_raw = p_cl = None
        if res.face_landmarks:
            lm = res.face_landmarks[0]
            p_raw = predict(gray, lm, w, h)
            p_cl = predict(clahe.apply(gray), lm, w, h)
        rows.append({"vid": vid_idx, "truth": seg["truth"],
                     "lighting": seg["lighting"], "glasses": seg["glasses"],
                     "raw": p_raw, "clahe": p_cl})
    cap.release()
    print("done:", video)

def stats(items, key, thr):
    sc = [r for r in items if r[key] is not None]
    if not sc:
        return "no data"
    y = np.array([r["truth"] for r in sc])
    pr = (np.array([r[key] for r in sc]) > thr).astype(int)
    acc = (pr == y).mean()
    rec = (pr[y == 1] == 1).mean() if (y == 1).any() else float("nan")
    fa = (pr[y == 0] == 1).mean() if (y == 0).any() else float("nan")
    return f"acc {acc:.0%} recall {rec:.0%} falsealarm {fa:.0%}"

n_closed = sum(r["truth"] == 1 for r in rows)
n_open = sum(r["truth"] == 0 for r in rows)
lines = [f"Frames used: {len(rows)} (closed={n_closed}, open={n_open})"]
for v in sorted(set(r["vid"] for r in rows)):
    vr = [r for r in rows if r["vid"] == v]
    lines.append(f"  video{v}: closed={sum(r['truth']==1 for r in vr)}, "
                 f"open={sum(r['truth']==0 for r in vr)}")
lines.append("")
for name, key in [("no CLAHE", "raw"), ("CLAHE", "clahe")]:
    lines.append(f"=== {name} ===")
    for thr in THRESHOLDS:
        lines.append(f"thr={thr}")
        lines.append(f"   ALL     : {stats(rows, key, thr)}")
        for v in sorted(set(r['vid'] for r in rows)):
            lines.append(f"   video{v}  : {stats([r for r in rows if r['vid']==v], key, thr)}")
        lines.append(f"   normal  : {stats([r for r in rows if r['lighting']=='normal'], key, thr)}")
        lines.append(f"   dim     : {stats([r for r in rows if r['lighting']=='dim'], key, thr)}")
    lines.append("")
text = "\n".join(lines)
print("\n" + text)
with open("my_results_sweep.txt", "w") as f:
    f.write(text + "\n")