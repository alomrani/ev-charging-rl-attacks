# ev-charging-rl-attacks

## Training

Code for training the detection model can be found in train_dnn.py

Code for training the RL agent can be found in train_rl.py

## Datasets

The training/validation/testing datasets for the RL agent can be found in the train/val/test_rl.pt files

The datasets for the detection models can be found in dnn_datasets directory

Each sample in these datasets contains a SoC sequence of an EV over the period of 24 hours (reported every 30 min). Therefore, each sample is of size 49 including the label (whether the sample is malicious or not)

## RL Environment

*charging_env.py*: Contains code for charging simulation

*reinforce_baselines.py*: Contains code for the Exponential baseline in policy gradient









