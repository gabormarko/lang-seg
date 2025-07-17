import os
from PIL import Image
import numpy as np
import argparse

def compute_mean_std(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    means = []
    stds = []
    for fname in image_files:
        img = np.array(Image.open(os.path.join(image_dir, fname)).convert('RGB')) / 255.0
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))
    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)
    print(f"Dataset mean: {mean}")
    print(f"Dataset std: {std}")
    return mean, std

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean and std for a folder of images.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input image directory')
    args = parser.parse_args()
    compute_mean_std(args.input_dir)
