"""
This module implements the data processing for training the SiamHCC model.
It prepares paired image samples and their similarity labels for the Siamese network.
"""

from PIL import Image
from torch.utils.data import Dataset
import random
import PIL.ImageOps
import numpy as np
import torch


class SiamHCCDataset(Dataset):
    """
    SiamHCCDataset: A dataset for training a Siamese network.

    This dataset is designed for training a Siamese network to determine whether 
    two images belong to the same class.

    Args:
        imageFolderDataset: A dataset object created using `torchvision.datasets.ImageFolder`, 
                            where `imageFolderDataset.imgs` is a list containing (image_path, class_index).
        transform: A set of preprocessing transformations, such as those defined in `torchvision.transforms` (default: None).
        should_invert: Whether to invert the image colors, typically used for grayscale images (default: True).
    """

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset  # Store the image dataset
        self.transform = transform  # Preprocessing transformations
        self.should_invert = should_invert  # Whether to apply color inversion

    def __getitem__(self, index):
        """
        Retrieves a sample pair (img0, img1) and its label.

        Returns:
            img0 (Tensor): The first image as a tensor.
            img1 (Tensor): The second image as a tensor.
            label (Tensor): A label indicating whether the two images belong to the same class (1) or different classes (0).
        """
        # Randomly select an image (img0)
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # Randomly determine whether img1 should be from the same class as img0
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            # Select an image from the same class
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:  # Ensure the class matches
                    break
        else:
            # Select an image from a different class
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:  # Ensure the class is different
                    break

        # Load images and convert them to RGB format
        img0 = Image.open(img0_tuple[0]).convert("RGB")
        img1 = Image.open(img1_tuple[0]).convert("RGB")

        # Apply color inversion if required
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        # Apply transformations if provided
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        # Return the processed images and their similarity label (1 for same class, 0 for different class)
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] == img0_tuple[1])], dtype=np.float32))
    

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.imageFolderDataset.imgs)
