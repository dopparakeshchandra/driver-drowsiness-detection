# Driver Drowsiness Detection

Detects whether a driver's eyes are awake or sleepy from eye images, using a CNN built with TensorFlow/Keras.

## Dataset
- 84,898 eye images, two classes: awake and sleepy
- Split 70/15/15: train 59,428, validation 12,735, test 12,735
- Classes are balanced

## Model
Simple CNN: 3 Conv2D + MaxPooling layers, Dropout 0.5, Dense 64, sigmoid output. Input 96x96. Trained for 10 epochs.

## Results (test set)

| Metric | Awake | Sleepy |
|---|---|---|
| Precision | 0.99 | 0.99 |
| Recall | 0.99 | 0.99 |
| F1-score | 0.99 | 0.99 |

- Test accuracy: 98.77%
- Sleepy recall: 98.6% (6,205 of 6,292 caught, 87 missed)

Confusion matrix:

| | Predicted awake | Predicted sleepy |
|---|---|---|
| Actually awake | 6373 | 70 |
| Actually sleepy | 87 | 6205 |

Full output: see `results.txt`.

## Limitations
Images from the same person may appear in train and test, so scores may be higher than on new people. Real-world testing on self-recorded videos is planned.

## Tech
Python, TensorFlow/Keras, OpenCV, scikit-learn 


Baseline vs your CNN:

Model	                   Test accuracy	Sleepy recall	Missed sleepy	False alarms
HOG + Logistic Regression	    95.24%	         95.6%	        276	            330
Simple CNN	                  98.77%	         98.6%	        87	            70
