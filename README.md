# Driver Drowsiness Detection

Real-time driver drowsiness detection from eye images using deep learning (TensorFlow/Keras). Three approaches are compared (a classical baseline, a custom CNN and MobileNetV2 transfer learning), and the best model powers a webcam app that raises an alert when the eyes stay closed. The model was also checked on my own webcam clips, which exposed a gap between the dataset and a real webcam and led to a fix (CLAHE contrast enhancement).

## Demo
Add a screenshot or GIF of the webcam app here (`demo.png` or `demo.gif`).

## Dataset
- **MRL Eye Dataset** (Media Research Lab, VSB-Technical University of Ostrava): https://mrl.cs.vsb.cz/eyedataset.html
- Kaggle copy used (awake/sleepy folders): [paste your Kaggle link here]
- 84,898 infrared eye images from 37 people, two classes: awake (open eyes) and sleepy (closed eyes)
- Split 70/15/15 (stratified, random_state=42), classes balanced

| Split | Awake | Sleepy | Total |
|---|---|---|---|
| Train | 30,066 | 29,362 | 59,428 |
| Validation | 6,443 | 6,292 | 12,735 |
| Test | 6,443 | 6,292 | 12,735 |

## Models
1. **Baseline:** HOG features (64x64 grayscale) + Logistic Regression
2. **Custom CNN:** 3 x (Conv2D + MaxPooling), Dropout 0.5, Dense 64, sigmoid output. Input 96x96, 10 epochs, Adam, binary cross-entropy
3. **MobileNetV2 (transfer learning):** ImageNet weights, 3 epochs with a frozen base, then 2 epochs fine-tuning the last 30 layers (learning rate 1e-5)

## Results on the dataset (test set, 12,735 images)

| Model | Accuracy | Sleepy recall | Sleepy precision | F1 (sleepy) | Missed sleepy | False alarms |
|---|---|---|---|---|---|---|
| HOG + Logistic Regression | 95.24% | 95.6% | 94.8% | 0.952 | 276 | 330 |
| **Custom CNN** | **98.77%** | **98.6%** | 98.9% | 0.99 | **87** | **70** |
| MobileNetV2 | 98.04% | 97.4% | 98.6% | 0.980 | 161 | 88 |

**Custom CNN confusion matrix**

| | Predicted awake | Predicted sleepy |
|---|---|---|
| Actually awake | 6,373 | 70 |
| Actually sleepy | 87 | 6,205 |

**Key findings**
- Both deep models clearly beat the baseline.
- The custom CNN cut total errors by about 74% (606 to 157) and missed sleepy cases by about 68% (276 to 87) compared with the baseline.
- The custom CNN slightly outperformed MobileNetV2, likely because the dataset is large and the images are small, uniform eye crops. MobileNetV2 was still improving at the end of fine-tuning.
- Recall for the sleepy class is the key safety metric, since a missed sleepy driver is the dangerous error.

Full outputs: `results.txt`, `baseline_results.txt`, `results_mobilenet.txt`

## Test on my own webcam clips
To check how the model works outside the dataset, I recorded two short webcam clips of myself (different lighting, with and without glasses), labelled the open and closed parts by hand (`labels.csv`), and tested the model on them: 152 frames, 50 closed and 102 open.

| Setting | Accuracy | Closed-eye recall | False alarms on open eyes |
|---|---|---|---|
| Original model input | 79% | 100% | 31% |
| + CLAHE contrast enhancement | 96% | 98% (49 of 50) | 5% |

- The dataset score (98.8%) did not carry over directly. The model was trained on infrared images, and my color webcam frames made it call many open eyes "closed".
- Adding CLAHE (contrast enhancement) to the eye image before prediction removed most of the false alarms while keeping nearly all closed-eye detections.
- Raising the decision threshold instead also reduced false alarms, but it started to miss closed eyes, which is the more dangerous error, so I kept the threshold at 0.5.
- Glasses made little difference in this test.
- **Limits:** one person, two clips, and only 5 closed frames in the dimmer clip. CLAHE was chosen after seeing these results, so the figure may be slightly optimistic. The recordings are not included in this repository.

## Real-time app
- OpenCV reads the webcam, MediaPipe Face Landmarker finds the eyes, CLAHE enhances the grayscale frame, and the custom CNN classifies each eye crop as open or closed.
- The two eye predictions are averaged. If the eyes stay closed for about 2 seconds, the app shows a warning and plays a beep. A short grace period (0.4 s) stops one flickering frame from resetting the timer, and normal blinks are too short to trigger the alert.
- Speed: about 15 FPS on a laptop CPU with CLAHE on (no GPU). The speed changes with the laptop and what else is running.

## How to run
```
pip install -r requirements.txt
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python split_data.py
python train_cnn.py
python evaluate.py
python realtime_app.py
```
Press `q` to quit the webcam app. The dataset and trained models are not stored in this repository, so `cnn_simple.keras` is created by `train_cnn.py`.

To repeat the own-video test, put your clips in `my_videos/`, edit `labels.csv` with the open and closed time ranges, then run `python test_videos_labeled.py` or `python test_sweep.py`.

## Project files
| File | Purpose |
|---|---|
| `split_data.py` | Stratified 70/15/15 split |
| `train_cnn.py`, `evaluate.py` | Train and evaluate the custom CNN |
| `baseline.py` | HOG + Logistic Regression baseline |
| `train_mobilenet.py`, `evaluate_mobilenet.py` | MobileNetV2 transfer learning |
| `realtime_app.py` | Real-time webcam detection with alert |
| `test_videos_labeled.py`, `test_sweep.py`, `labels.csv` | Test on my own labelled webcam clips (with and without CLAHE, different thresholds) |
| `web_cam_test.py` | Webcam check |

## Limitations
- The dataset has 37 people and the split was random, so images of the same person may appear in train, validation and test. The dataset scores may be higher than on unseen people.
- The model was trained on infrared images, while a normal webcam gives color images. Real-world accuracy can be lower than the dataset score, as the own-clip test showed.
- The own-clip test is small (one person, 152 frames), so it is an indication and not a full evaluation.

## Roadmap
- [x] Dataset split and baseline
- [x] Custom CNN and MobileNetV2 comparison
- [x] Real-time webcam app with alert
- [x] FPS measurement (about 15 FPS on CPU with CLAHE)
- [x] Small test on my own webcam clips, with a CLAHE fix
## Tech
Python, TensorFlow/Keras, OpenCV, MediaPipe, scikit-learn, scikit-image, NumPy
