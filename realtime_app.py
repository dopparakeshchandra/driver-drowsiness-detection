import cv2, time, threading
import numpy as np
import mediapipe as mp
import tensorflow as tf 
import winsound
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

model = tf.keras.models.load_model("cnn_simple.keras")


options = vision.FaceLandmarkerOptions(
    base_options = mp_python.BaseOptions(model_asset_path = 'face_landmarker.task'),
    running_mode = vision.RunningMode.VIDEO,
    num_faces = 1)
landmarker = vision.FaceLandmarker.create_from_options(options) 

LEFT = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT = [362, 263, 387, 386, 385, 373, 374, 380]
THRESH = 0.5
ALERT_SECONDS = 2.0

def crop_eye(gray, Lm, idx, w, h):
    xs = [Lm[i].x * w for i in idx]
    ys = [Lm[i].y * h for i in idx]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = (max(xs) - min(xs)) * 1.6
    x1, x2 = int(max(cx - side / 2,0 )), int(min(cx + side / 2, w))
    y1, y2 = int(max(cy - side / 2, 0)), int(min(cy + side / 2, h))
    eye = gray[y1:y2, x1:x2]
    if eye.size == 0:
        return None, (x1, y1, x2, y2)
    eye = cv2.resize(eye, (96, 96))
    eye = np.stack([eye] * 3, axis = -1).astype("float32")
    return eye, (x1, y1, x2, y2)

def beep():
    winsound.Beep(1200, 400)
    
cap = cv2. VideoCapture(0)
closed_since, last_beep, prev = None, 0, time.time()
fps_values = []
start = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    h,w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image= mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)
    timestamp_ms = int((time.time() - start) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    status, color, p = "NO FACE", (200,200, 200), None
    now = time.time()
    
    if result.face_landmarks:
        Lm = result.face_landmarks[0] 
        eyes, boxes = [], []
        for idx in (LEFT, RIGHT):
            e, box = crop_eye(gray, Lm, idx, w, h)
            if e is not None:
                eyes.append(e)
                boxes.append(box)
                
        if eyes:
            p = float(model(np.array(eyes), training = False).numpy().mean())
            closed = p > THRESH
            status = "EYES CLOSED" if closed else "EYES OPEN" 
            color = (0, 0, 255) if closed else (0,200, 0)
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
            if closed:
                closed_since = closed_since or now
                if now - closed_since >= ALERT_SECONDS: 
                    cv2.putText(frame, "WAKE UP! DROWSINESS ALERT", (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
                    if now - last_beep > 1.0:
                        threading.Thread(target = beep, daemon = True).start()
                        last_beep = now
            else:
                closed_since = None
    else:
        closed_since = None
        
    fps = 1 / max(now - prev, 1e-6)
    prev = now
    fps_values.append(fps)
    
    label = f"{status}" + (f" p(sleepy)={p:.2f}" if p is not None else "")
    cv2.putText(frame, label, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()
print(f"Average FPS: {np.mean(fps_values[10:]):.1f}")

    
                  
    
    