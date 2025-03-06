"""
This script handles dataset loading, model training, and loss monitoring.
"""

import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import SiamHCCDataset
from SiamHCC import SiamHCC


# Set training parameters
training_dir = r"HCCE"          # Path to the training directory containing images
train_batch_size = 16           # Batch size for training
train_number_epochs = 200       # Number of epochs to train
input_shape = [64, 64]          # Desired input image size (width, height)

# Define image transformation: resize images and convert them to tensors
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Create an ImageFolder dataset from the training directory
folder_dataset = dset.ImageFolder(root=training_dir)
# Initialize the custom SiamHCC dataset with the image folder dataset
siamese_dataset = SiamHCCDataset(
    imageFolderDataset=folder_dataset,
    transform=transform,
    should_invert=False  # Do not invert image colors
)
# Create a DataLoader for batching and shuffling the dataset
train_dataloader = DataLoader(
    siamese_dataset,
    shuffle=True,
    num_workers=0,
    batch_size=train_batch_size
)

# Initialize the SiamHCC network
net = SiamHCC()
# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
net.to(device)

# Define the loss function (BCEWithLogitsLoss is used for binary classification tasks)
criterion = torch.nn.BCEWithLogitsLoss()
# Define the optimizer (Adam) with a learning rate of 0.001
optimizer = torch.optim.Adam(net.parameters(), 0.001, betas=(0.9, 0.999))

# Variables to track training progress
counter = []
loss_history = []
iteration_number = 0

if __name__ == '__main__':
    # Main training loop
    for epoch in range(0, train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            # Retrieve a pair of images and their similarity label
            img0, img1, label = data
            # Transfer images and labels to the selected device (GPU/CPU)
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            
            # Zero the gradients for the optimizer
            optimizer.zero_grad()
            # Forward pass: compute network output given the image pair
            output = net(img0, img1)
            # Compute the loss between predicted similarity and ground truth
            loss_contrastive = criterion(output, label)
            # Backward pass: compute gradients
            loss_contrastive.backward()
            # Update network weights based on computed gradients
            optimizer.step()
            
            # Every 10 iterations, print the current training status
            if i % 10 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    
    # Plot the training loss over iterations
    plt.plot(counter, loss_history)
    plt.show()
    
    # Save the trained model weights
    torch.save(net.state_dict(), 'weights/SiamHCC.pkl')
