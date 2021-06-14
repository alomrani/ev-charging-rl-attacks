# Program Manual


## Training

`train_dnn.py`: code for training the detection model.

`train_rl.py`: code for training the RL agent.

`options.py`: Description of arguments for training the RL agent.

`options1.py`: Description of arguments for training the detection model.

`soc_dataset.py`: PyTorch dataset class for defining how to retrieve samples from the dataset.

To train a detection model on a dataset, use the following command and input required arguments:

`python train_ddn.py --lr_model LEARNING_RATE --lr_decay LEARNING_RATE_DECAY --n_epochs NUM_EPOCHS --batch_size BATCH_SIZE`
 
See `options1.py` for other arguments that can be specified.

To train an RL agent, use the following command and input required arguments:

`python train_rl.py --lr_model LEARNING_RATE --lr_decay LEARNING_RATE_DECAY --n_epochs NUM_EPOCHS --batch_size BATCH_SIZE --exp_beta EXP_BETA`

See `options.py` for other arguments that can be specified.

## RL Environment

`charging_env.py`: Contains code for charging simulation.

`reinforce_baselines.py`: Contains code for the Exponential baseline in policy gradient.

## Attacks policies

*DNNAgent.py*: Model for the adversarial RL agent.
*spoof_agentX.py*: Model for agent which follows synthetic Attack strategy X (See paper for synthetic attacks considered).

## Detection Model

*DetectionModelDNN.py*: Defined the DNN architecture of the detection model.

# Datasel Manual

The training/validation/testing datasets for the RL agent can be found in the `rl_datasets` directory.

The datasets for the detection models can be found in `dnn_datasets` directory.

Each sample in these datasets contains a SoC sequence of an EV over the period of 24 hours (reported every 30 min). Therefore, **each sample is of size 49 including the label i.e. whether the sample is malicious or not**.



Feedback is welcome.





