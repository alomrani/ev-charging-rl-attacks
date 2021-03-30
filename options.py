
import os
import time
import argparse
import torch


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="RL agent to generate attacks for training a detection model."
    )

    # Data
    parser.add_argument(
        "--problem",
        type=str,
        default="obm",
        help="Problem: 'obm', 'e-obm', 'adwords' or 'displayads'",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=536,
        help="Number of instances per batch during training",
    )

    parser.add_argument(
        "--val_size",
        type=float,
        default=1072.,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="rl_val.pt",
        help="Dataset file to use for validation",
    )

    parser.add_argument(
        "--train_dataset",
        type=str,
        default="rl_train.pt",
        help="Dataset file to use for training",
    )

    parser.add_argument(
        "--dataset_size", type=int, default=11792, help="Dataset size for training",
    )
    
    parser.add_argument(
        "--num_cars", type=int, default=30, help="Number of cars in the simulation for training",
    )
    
    parser.add_argument(
        "--epsilon", type=float, default=0.6, help="Epsilon parameter for charging coordinator",
    )
    parser.add_argument(
        "--lamb", type=int, default=20, help="Poisson rate for number of arriving cars",
    )
    parser.add_argument(
        "--total_power", type=float, default=1500, help="Total power capacity of charging station at each timestep",
    )
    parser.add_argument(
        "--battery_capacity", type=float, default=200, help="Battery capacity of vehicle",
    )
    
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=264,
        help="Dimension of hidden layers in Enc/Dec",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=48,
        help="Number of timesteps in charging simulation",
    )

    # Training
    parser.add_argument(
        "--lr_model",
        type=float,
        default=0.0004,
        help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.99, help="Learning rate decay per epoch"
    )

    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="The number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--exp_beta",
        type=float,
        default=0.7,
        help="Exponential moving average baseline decay (default 0.8)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.01,
        help="hyperparameter for regularization in reward function",
    )
    parser.add_argument(
        "--regularize",
        action="store_true",
        help="Set this value to add regularization to the reward function",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Set this value to only evaluate model on a specific graph size",
    )
    parser.add_argument(
        "--train_seed",
        action="store_true",
        help="train agent on different seeds and plot avg rewards",
    )

    parser.add_argument(
        "--eval_output", type=str, default=".", help="path to output evaulation plots",
    )
    parser.add_argument(
        "--load_path", type=str, default=".", help="path to model's parameters",
    )
    # Misc
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Set this to true if you want to tune the hyperparameters",
    )

    parser.add_argument(
        "--output_dir", default="outputs", help="Directory to write output models to"
    )

    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=0,
        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints",
    )


    parser.add_argument(
        "--save_dir", help="Path to save the checkpoints",
    )


    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    if opts.use_cuda:
        opts.device = "cuda"
    else:
        opts.device = "cpu"
    opts.run_name = "{}_{}".format("run", time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(opts.output_dir, opts.run_name)

    assert (
        opts.dataset_size % opts.batch_size == 0
    ), "Epoch size must be integer multiple of batch size!"
    return opts
