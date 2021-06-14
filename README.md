# 1. Program Manual


## Training

`train_dnn.py`: code for training the detection model.

`train_rl.py`: code for training the RL agent.

`options.py`: Description of arguments for training the RL agent.

`options1.py`: Description of arguments for training the detection model.

`soc_dataset.py`: PyTorch dataset class for defining how to retrieve samples from the dataset.

To train a detection model on a dataset, use the following command and input required arguments:

`python train_ddn.py --lr_model LEARNING_RATE --lr_decay LEARNING_RATE_DECAY --n_epochs NUM_EPOCHS --batch_size BATCH_SIZE`
 
See `options1.py` for other arguments that can be specified.

Outputs of a run will be saved to `outputs/dnn/run_X`

To train an RL agent with gamma=0, use the following command and input required arguments:

`python train_rl.py --lr_model LEARNING_RATE --lr_decay LEARNING_RATE_DECAY --n_epochs NUM_EPOCHS --batch_size BATCH_SIZE --exp_beta EXP_BETA`

To train an RL agent with gamma regularization, add the arguments `--regularize --gamma GAMMA`.

See `options.py` for other arguments that can be specified.

Outputs of a run will saved to `outputs/[NUM_CARS]_[GAMMA]/run_X`

## Evaluation

To evaluate a trained detection model on the test dataset, run the following and add the path to model parameters:

`python train_dnn.py --eval_only --load_path PATH_TO_TRAINED_MODEL`

To test a trained RL agent in the charging simulation, run the following with the path to agent's parameters:

`python train_rl.py --eval_only --load_path PATH_TO_TRAINED_AGENT`, add the arguments `--regularize --gamma GAMMA` if agent was trained with gamma regularization.

To test a synthetic attack strategy:
`python train_rl.py --eval_only --attack_model attackX`

Where `X` represents the synthetic attack type and can be any of `1-4`.

To test the detection accuracy of a DNN model on RL agent attacks:

`python train_rl.py --eval_detect --load_path PATH_TO_TRAINED_AGENT --load_path2 PATH_TO_DETECTION_MODEL --gamma GAMMA --regularize`

To test detection accuracy on a synthetic attack:

`python train_rl.py --eval_detect --load_path PATH_TO_TRAINED_AGENT --attack_model attackX`

## RL Environment

`charging_env.py`: Contains code for charging simulation.

`reinforce_baselines.py`: Contains code for the Exponential baseline in policy gradient.

## Attacks policies

`DNNAgent.py`: Model for the adversarial RL agent.
`spoof_agentX.py`: Model for agent which follows synthetic Attack strategy X (See paper for synthetic attacks considered).

## Detection Model

`DetectionModelDNN.py`: Defined the DNN architecture of the detection model.


# 2. Datasel Manual

The training/validation/testing datasets for the RL agent can be found in the `rl_datasets` directory:
- Files with name formal `dataset_X_syn.py` correspong to dataset with both synthetic and intelligent attacks.
- Files with format `dataset_X.pt` correspond to dataset with intelligent attacks only.

The datasets for the detection models can be found in `dnn_datasets` directory.

Each sample in these datasets contains a SoC sequence of an EV over the period of 24 hours (reported every 30 min). Therefore, **each sample is of size 49 including the label i.e. whether the sample is malicious or not**.



Feedback is welcome.





