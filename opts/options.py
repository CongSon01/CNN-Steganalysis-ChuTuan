"""This module provides method to enter various input to the model training."""
import argparse


def arguments() -> str:
    """This function returns arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cover_path",
        default="./data/train/cover",
    )
    parser.add_argument(
        "--stego_path",
        default="./data/train/stego",
    )
    parser.add_argument(
        "--valid_cover_path",
        default="./data/valid/cover",
    )
    parser.add_argument(
        "--valid_stego_path",
        default="./data/valid/stego",
    )
    parser.add_argument("--checkpoints_dir", default="./checkpoints/")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=13)
    parser.add_argument("--val_size", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.001)

    opt = parser.parse_args()
    return opt
