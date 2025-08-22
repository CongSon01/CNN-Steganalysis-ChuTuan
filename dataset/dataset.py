"""This module provide the data sample for training."""

import os
from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
import glob

import imageio as io

from opts.options import arguments

opt = arguments()
# pylint: disable=E1101
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=E1101


class DatasetLoad(Dataset):
    """This class returns the data samples."""

    def __init__(
        self,
        cover_path: str,
        stego_path: str,
        size: int,
        transform: Tuple = None,
    ) -> None:
        """Constructor.

        Args:
            cover_path (str): path to cover images.
            stego_path (str): path to stego images.
            size (int): no. of images in any of (cover / stego) directory for
              training.
            transform (Tuple, optional): _description_. Defaults to None.
        """
        self.cover = cover_path
        self.stego = stego_path
        self.transforms = transform
        self.data_size = size
        
        # Get list of available files
        cover_files = sorted([f for f in os.listdir(cover_path) if f.lower().endswith(('.pgm', '.bmp', '.png', '.jpg', '.jpeg'))])
        stego_files = sorted([f for f in os.listdir(stego_path) if f.lower().endswith(('.pgm', '.bmp', '.png', '.jpg', '.jpeg'))])
        
        # Make sure we have enough files
        if len(cover_files) < size or len(stego_files) < size:
            raise ValueError(f"Not enough files. Need {size}, but found cover: {len(cover_files)}, stego: {len(stego_files)}")
        
        # Store file lists (use only first 'size' files)
        self.cover_files = cover_files[:size]
        self.stego_files = stego_files[:size]

    def __len__(self) -> int:
        """returns the length of the dataset."""
        return self.data_size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Returns the (cover, stego) pairs for training.

        Args:
            index (int): a random int value in range (0, len(dataset)).
        Returns:
            Tuple[Tensor, Tensor]: cover and stego pair.
        """
        # Use the file lists instead of constructing names
        cover_filename = self.cover_files[index]
        stego_filename = self.stego_files[index]
        
        cover_img = io.imread(os.path.join(self.cover, cover_filename))
        stego_img = io.imread(os.path.join(self.stego, stego_filename))
        # pylint: disable=E1101
        label1 = torch.tensor(0, dtype=torch.long).to(device)
        label2 = torch.tensor(1, dtype=torch.long).to(device)
        # pylint: enable=E1101
        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)
            sample = {"cover": cover_img, "stego": stego_img}
        sample["label"] = [label1, label2]
        return sample
