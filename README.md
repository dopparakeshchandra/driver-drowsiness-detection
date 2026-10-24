# Driver Drowsiness Detection

Real-time driver drowsiness detection from eye images using deep learning (TensorFlow/Keras). Three approaches are compared (a classical baseline, a custom CNN and MobileNetV2 transfer learning), and the best model powers a webcam app that raises an alert when the eyes stay closed.

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

## Results (test set, 12,735 images)

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

## Real-time app
- OpenCV reads the webcam, MediaPipe Face Landmarker finds the eyes, and the custom CNN classifies each eye crop as open or closed.
- If the eyes stay closed for about 2 seconds, the app shows a warning and plays a beep.
- Speed: about 9.5 FPS on a laptop CPU (no GPU).

## How to run
```
pip install -r requirements.txt
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
python split_data.py
python train_cnn.py
python evaluate.py
python realtime_app.py
```
Press `q` to quit the webcam app. The dataset and trained models are not stored in this repository.

## Project files
| File | Purpose |
|---|---|
| `split_data.py` | Stratified 70/15/15 split |
| `train_cnn.py`, `evaluate.py` | Train and evaluate the custom CNN |
| `baseline.py` | HOG + Logistic Regression baseline |
| `train_mobilenet.py`, `evaluate_mobilenet.py` | MobileNetV2 transfer learning |
| `realtime_app.py` | Real-time webcam detection with alert |
| `web_cam_test.py` | Webcam check |

## Limitations
- The dataset has 37 people and the split was random, so images of the same person may appear in train, validation and test. These scores may be higher than on unseen people.
- The model was trained on infrared eye images, while a normal webcam gives color images. Real-world accuracy can differ from the test score.
- Real-time performance so far is only checked informally on one person.

## Roadmap
- [x] Dataset split and baseline
- [x] Custom CNN and MobileNetV2 comparison
- [x] Real-time webcam app with alert
- [x] FPS measurement (about 9.5 FPS on CPU)
- [ ] Testing on self-recorded videos (different lighting, glasses) with error analysis
- [ ] Subject-wise split for a stricter evaluation
- [ ] Deployed demo and demo video

## Tech
Python, TensorFlow/Keras, OpenCV, MediaPipe, scikit-learn, scikit-image, NumPy
