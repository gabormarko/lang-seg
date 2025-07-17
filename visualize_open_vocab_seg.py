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
    parser.add_argument('--output_path', type=str, default='/home/neural_fields/Unified-Lift-Gabor/cuda_project_image_to_sparse_voxel/vis_open_voc', help='Optional path to save the segmentation mask image')
    parser.add_argument('--original_image', type=str, default=None, help='Optional path to original image for overlay')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='CLIP model to use (e.g., ViT-L/14)')
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

    # Ensure both features and text_features are float32 for matmul
    features = features.float()
    text_features = text_features.float()

    # Compute similarity
    similarity = features @ text_features.T  # [H*W, num_labels]
    pred = similarity.argmax(dim=1).reshape(H, W).cpu().numpy()  # [H, W]

    # Visualization
    import os
    import matplotlib.patches as mpatches

    def get_default_filename():
        if args.original_image:
            base = os.path.splitext(os.path.basename(args.original_image))[0]
            return f"segmentation_{base}.png"
        else:
            return "segmentation_output.png"

    if args.output_path:
        # If output_path is a directory, append default filename
        if os.path.isdir(args.output_path):
            output_path = os.path.join(args.output_path, get_default_filename())
        # If output_path ends with .png or .jpg, treat as file
        elif args.output_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_path = args.output_path
        else:
            # If not a directory and not a file, treat as directory and append filename
            output_path = os.path.join(args.output_path, get_default_filename())
    else:
        output_path = get_default_filename()

    # Overlay visualization with legend
    fig, ax = plt.subplots(figsize=(8, 8))
    if args.original_image:
        orig = Image.open(args.original_image).convert('RGB').resize((W, H), resample=Image.BILINEAR)
        ax.imshow(orig)
        im = ax.imshow(pred, cmap='tab10', alpha=0.5, vmin=0, vmax=len(args.labels)-1)
    else:
        im = ax.imshow(pred, cmap='tab10', vmin=0, vmax=len(args.labels)-1)
    ax.set_title(f"Open-vocab Segmentation: {', '.join(args.labels)}")
    ax.axis('off')

    # Add legend for labels
    handles = [mpatches.Patch(color=plt.cm.tab10(i), label=label) for i, label in enumerate(args.labels)]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Labels")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved segmentation visualization to {output_path}")
    plt.close(fig)

    # Save mask-only image (no overlay, no axis, no title, no legend)
    mask_only_path = output_path.replace('.png', '_mask.png')
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(pred, cmap='tab10', vmin=0, vmax=len(args.labels)-1)
    ax2.axis('off')
    fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig2.savefig(mask_only_path, bbox_inches='tight', pad_inches=0, dpi=150)
    print(f"Saved mask-only segmentation to {mask_only_path}")
    plt.close(fig2)

if __name__ == "__main__":
    main()
