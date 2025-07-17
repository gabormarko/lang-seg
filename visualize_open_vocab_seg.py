import numpy as np
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image
import argparse

# --- Script to visualize open-vocabulary segmentation from saved per-pixel features and text queries ---
def main():
    parser = argparse.ArgumentParser(description="Visualize open-vocabulary segmentation from per-pixel features and text labels.")
    parser.add_argument('--features_path', type=str, required=True, help='Path to .npy file with per-pixel features')
    parser.add_argument('--labels', type=str, nargs='+', required=True, help='List of text labels to segment')
    parser.add_argument('--output_path', type=str, default=None, help='Optional path to save the segmentation mask image')
    parser.add_argument('--original_image', type=str, default=None, help='Optional path to original image for overlay')
    parser.add_argument('--clip_model', type=str, default='ViT-L/14', help='CLIP model to use (e.g., ViT-L/14)')
    args = parser.parse_args()

    # Load per-pixel features
    features = np.load(args.features_path)  # [C, H, W]
    if features.dtype == np.float16:
        features = features.astype(np.float32)
    features = torch.from_numpy(features)  # [C, H, W]
    C, H, W = features.shape

    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load(args.clip_model, device=device)

    # Encode text labels
    text_tokens = clip.tokenize(args.labels).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)  # [num_labels, D]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Prepare image features
    features = features.permute(1, 2, 0).reshape(-1, C).to(device)  # [H*W, C]
    features = features / features.norm(dim=-1, keepdim=True)

    # Compute similarity
    similarity = features @ text_features.T  # [H*W, num_labels]
    pred = similarity.argmax(dim=1).reshape(H, W).cpu().numpy()  # [H, W]

    # Visualization
    plt.figure(figsize=(8, 8))
    if args.original_image:
        orig = Image.open(args.original_image).convert('RGB').resize((W, H), resample=Image.BILINEAR)
        plt.imshow(orig)
        plt.imshow(pred, cmap='tab10', alpha=0.5)
    else:
        plt.imshow(pred, cmap='tab10')
    plt.title(f"Open-vocab Segmentation: {', '.join(args.labels)}")
    plt.axis('off')
    if args.output_path:
        plt.savefig(args.output_path, bbox_inches='tight', dpi=150)
        print(f"Saved segmentation visualization to {args.output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
