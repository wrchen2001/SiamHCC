import argparse
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import torch
from PIL import Image
from datasets import SiamHCCDataset
from SiamHCC import SiamHCC

def visualize_pair(img_pair, similarity, save_path=None):
    """
    Visualize image pair and similarity score
    
    Args:
        img_pair: Concatenated image tensor (Tensor)
        similarity: Similarity score (float)
        save_path: Path to save visualization (optional)
    """
    grid = torchvision.utils.make_grid(img_pair).cpu()
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.title(f"Similarity: {similarity:.4f}", pad=20)
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def load_image(image_path, transform):
    """Load and preprocess single image"""
    try:
        img = Image.open(image_path).convert("RGB")
        return transform(img)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading image {image_path}: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SiamHCC Character Similarity Evaluation')
    parser.add_argument('--img1', type=str, required=True, help='Path to first character image')
    parser.add_argument('--img2', type=str, required=True, help='Path to second character image')
    parser.add_argument('--weights', type=str, default='weights/CCSnet.pkl',
                      help='Path to pretrained weights')
    parser.add_argument('--output', type=str, help='Path to save result visualization')
    args = parser.parse_args()

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Initialize model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = SiamHCC().to(device)
    
    try:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Successfully loaded weights from {args.weights}")
    except Exception as e:
        raise RuntimeError(f"Error loading model weights: {str(e)}")

    model.eval()

    # Load and preprocess images
    try:
        img1 = load_image(args.img1, transform)
        img2 = load_image(args.img2, transform)
    except Exception as e:
        print(e)
        return

    # Prepare input tensor
    batch = torch.stack([
        img1.unsqueeze(0).to(device),
        img2.unsqueeze(0).to(device)
    ])

    # Perform inference
    with torch.no_grad():
        similarity = torch.sigmoid(model(*batch)).item()
    
    print(similarity)

    # Visualize results
    visualize_pair(
        torch.stack([img1.cpu(), img2.cpu()]), 
        similarity,
        save_path=args.output
    )

if __name__ == "__main__":
    main()