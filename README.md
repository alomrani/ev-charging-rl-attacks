# ev-charging-rl-attacks

## Training

*train_dnn.py*: code for training the detection model.

*train_rl.py*: code for training the RL agent.

*options.py*: arguments for training the RL agent.

*options1.py*: arguments for training the detection model.

## Datasets

The training/validation/testing datasets for the RL agent can be found in the *rl_datasets* directory

The datasets for the detection models can be found in *dnn_datasets* directory

Each sample in these datasets contains a SoC sequence of an EV over the period of 24 hours (reported every 30 min). Therefore, **each sample is of size 49 including the label i.e. whether the sample is malicious or not**

*soc_dataset.py*: PyTorch dataset class for defining how to retrieve samples from the dataset

## RL Environment

*charging_env.py*: Contains code for charging simulation

*reinforce_baselines.py*: Contains code for the Exponential baseline in policy gradient









