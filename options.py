
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
        default=10,
        help="Number of instances per batch during training",
    )

    )
    #    parser.add_argument(
    #        "--epoch_size",
    #        type=int,
    #        default=100,
    #        help="Number of instances per epoch during training",
    #    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=1000,
        help="Number of instances used for reporting validation performance",
    )
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="dataset/val",
        help="Dataset file to use for validation",
    )

    parser.add_argument(
        "--train_dataset",
        type=str,
        default="dataset/train",
        help="Dataset file to use for training",
    )

    parser.add_argument(
        "--dataset_size", type=int, default=1000, help="Dataset size for training",
    )




    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=16,
        help="Dimension of hidden layers in Enc/Dec",
    )



    # Training
    parser.add_argument(
        "--lr_model",
        type=float,
        default=1e-3,
        help="Set the learning rate for the actor network",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.99, help="Learning rate decay per epoch"
    )

    parser.add_argument(
        "--n_epochs", type=int, default=1000, help="The number of epochs to train"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed to use")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--exp_beta",
        type=float,
        default=0.8,
        help="Exponential moving average baseline decay (default 0.8)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Baseline to use: 'rollout', 'critic' or 'exponential'. Defaults to no baseline.",
    )
 


  
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Set this value to only evaluate model on a specific graph size",
    )


    #    parser.add_argument(
    #        "--eval_family",
    #        action="store_true",
    #        help="Set this to true if you evaluating the model over a family of graphs",
    #    )
    parser.add_argument(
        "--eval_output", type=str, default=".", help="path to output evaulation plots",
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
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(opts.output_dir, opts.model, opts.run_name)

    assert (
        opts.dataset_size % opts.batch_size == 0
    ), "Epoch size must be integer multiple of batch size!"
    return opts
