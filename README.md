# ev-charging-rl-attacks


Code for training the detection model can be found in train_dnn.py

Code for training the RL agent can be found in train_rl.py

# Datasets

The training/validation/testing datasets for the RL agent can be found in the train/val/test_rl.pt files

Each sample in these datasets contains a benign SoC sequence of an EV over the period of 48 hours i.e each sample is of size 48

The training/validation/testing datasets for the detection models can be found in detection_train/val/test.pt files


