# monkeyTCN
Temporal Convolutional Network to predict behavioral data (EMG / kinematics) from neural recordings.

## Usage
`python *_test.py`
This command will train a model on monkey neural data to decode the EMG activity of a sample muscle, with our default parameters, save the model, and print crossvalidated VAF of the reconstructed signal. Default parameters can be seen in `*_test.py`.


### Compatibility
Only works on Python 2.x, although most of it is Python 3.x compatible.
