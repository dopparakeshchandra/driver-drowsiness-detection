# Driver Drowsiness Detection

Classifies eye images as **awake** or **sleepy** using deep learning (TensorFlow/Keras). Three approaches are compared: a classical baseline, a custom CNN and MobileNetV2 transfer learning. A real-time webcam alert system is in progress.

## Dataset
- 84,898 cropped eye images, two classes: awake and sleepy
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
- The custom CNN slightly outperformed MobileNetV2 here, likely because the dataset is large and the images are small, uniform eye crops. MobileNetV2 was still improving at the end of fine-tuning.
- Recall for the sleepy class is the key safety metric, since a missed sleepy driver is the dangerous error.

Full outputs: `results.txt`, `baseline_results.txt`, `results_mobilenet.txt`

## Project files
| File | Purpose |
|---|---|
| `split_data.py` | Stratified 70/15/15 split |
| `train_cnn.py`, `evaluate.py` | Train and evaluate the custom CNN |
| `baseline.py` | HOG + Logistic Regression baseline |
| `train_mobilenet.py`, `evaluate_mobilenet.py` | MobileNetV2 transfer learning |

## How to run
```
pip install -r requirements.txt
python split_data.py
python train_cnn.py
python evaluate.py
```

## Limitations
- Images from the same person may appear in train, validation and test, so these scores may be higher than on unseen people.
- Evaluation so far is on a public dataset only. Testing on self-recorded videos in different lighting, with and without glasses, is planned.

## Roadmap
- [ ] Real-time webcam app (OpenCV + MediaPipe) with an alert after about 2 seconds of closed eyes
- [ ] FPS measurement
- [ ] Testing on self-recorded videos and error analysis
- [ ] Deployed demo and demo video

## Tech
Python, TensorFlow/Keras, OpenCV, scikit-learn, scikit-image, NumPy
