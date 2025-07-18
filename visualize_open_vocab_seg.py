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
    print(f"[DEBUG] Loaded features from {args.features_path}, shape: {features.shape}, dtype: {features.dtype}")
    if features.dtype == np.float16:
        features = features.astype(np.float32)
    features = torch.from_numpy(features)  # [C, H, W]
    C, H, W = features.shape
    print(f"[DEBUG] Features tensor shape after conversion: {features.shape}")

    from modules.lseg_module import LSegModule
    checkpoint_path = 'checkpoints/demo_e200.ckpt'
    module = LSegModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        data_path='/tmp',
        dataset='lerf',
        backbone='clip_vitl16_384',
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=255,
        dropout=0.0,
        scale_inv=False,
        augment=False,
        no_batchnorm=False,
        widehead=True,
        widehead_hr=False,
        map_locatin="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )
    # Use .net if available, else use module directly
    try:
        from encoding.models.sseg import BaseNet
        if hasattr(module, 'net') and isinstance(module.net, BaseNet):
            model = module.net
        else:
            model = module
    except ImportError:
        model = module
    model = model.eval().cpu()

    labels = args.labels
    print(f"[DEBUG] Label list used for visualization: {labels}")
    features = features.unsqueeze(0)  # [1, C, H, W]
    print(f"[DEBUG] Features shape before model: {features.shape}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    features = features.to(device)
    with torch.no_grad():
        # Use the model's post-processing to get segmentation from extracted features
        if hasattr(model, 'net') and hasattr(model.net, 'project_features_to_labels'):
            seg_output = model.net.project_features_to_labels(features, labelset=labels, device=device)
        elif hasattr(model, 'project_features_to_labels'):
            seg_output = model.project_features_to_labels(features, labelset=labels, device=device)
        else:
            raise AttributeError("Neither model nor model.net has project_features_to_labels method.")
        print(f"[DEBUG] seg_output shape: {seg_output.shape if hasattr(seg_output, 'shape') else [o.shape for o in seg_output]}")
        pred = torch.argmax(seg_output[0], dim=0).cpu().numpy()  # [H, W]
        print(f"[DEBUG] Predicted mask shape: {pred.shape}")

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

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, preprocess = clip.load(args.clip_model, device=device)
    # Use lseg_app.py palette and legend logic
    def get_new_pallete(num_cls):
        n = num_cls
        pallete = [0]*(n*3)
        for j in range(0,n):
            lab = j
            pallete[j*3+0] = 0
            pallete[j*3+1] = 0
            pallete[j*3+2] = 0
            i = 0
            while (lab > 0):
                pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                i = i + 1
                lab >>= 3
        return pallete

    def get_new_mask_pallete(npimg, new_palette, out_label_flag=False, labels=None):
        out_img = Image.fromarray(npimg.squeeze().astype('uint8'))
        out_img.putpalette(new_palette)
        patches = []
        if out_label_flag:
            assert labels is not None
            u_index = np.unique(npimg)
            for i, index in enumerate(u_index):
                label = labels[index]
                cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
                red_patch = mpatches.Patch(color=cur_color, label=label)
                patches.append(red_patch)
        return out_img, patches

    # Upsample mask if needed
    if args.original_image:
        orig = Image.open(args.original_image).convert('RGB')
        orig_w, orig_h = orig.size
        print(f"[DEBUG] Loaded original image from {args.original_image}, shape: {orig.size}")
        print(f"[DEBUG] Model mask shape before upsampling: {pred.shape}")
        import torch.nn.functional as F
        pred_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
        print(f"[DEBUG] pred_tensor shape for upsampling: {pred_tensor.shape}")
        up_pred = F.interpolate(pred_tensor, size=(orig_h, orig_w), mode='nearest')[0,0].numpy().astype(int)
        print(f"[DEBUG] Upsampled mask shape: {up_pred.shape}")
        mask_for_vis = up_pred
        overlay_img = orig
    elif os.path.exists(args.features_path.replace('.JPG.npy', '_preproc.png')):
        preproc_img_path = args.features_path.replace('.JPG.npy', '_preproc.png')
        preproc_img = Image.open(preproc_img_path).convert('RGB')
        print(f"[DEBUG] Loaded preprocessed image from {preproc_img_path}, shape: {preproc_img.size}")
        print(f"[DEBUG] Model mask shape for preproc overlay: {pred.shape}")
        mask_for_vis = pred
        overlay_img = preproc_img
    else:
        print(f"[DEBUG] No overlay image used, only mask.")
        print(f"[DEBUG] Model mask shape for mask-only: {pred.shape}")
        mask_for_vis = pred
        overlay_img = None

    new_palette = get_new_pallete(len(labels))
    mask_img, legend_patches = get_new_mask_pallete(mask_for_vis, new_palette, out_label_flag=True, labels=labels)
    seg_rgba = mask_img.convert("RGBA")

    fig, ax = plt.subplots(figsize=(8, 8))
    if overlay_img is not None:
        ax.imshow(overlay_img)
        ax.imshow(seg_rgba, alpha=0.7)
    else:
        ax.imshow(seg_rgba)
    # Show only top 5 labels in title, then '...'
    if len(labels) > 5:
        title_labels = ', '.join(labels[:5]) + ', ...'
    else:
        title_labels = ', '.join(labels)
    ax.set_title(f"Open-vocab Segmentation: {title_labels}")
    ax.axis('off')
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Labels")


    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved segmentation visualization to {output_path}")
    plt.close(fig)

    # Save mask-only image (no overlay, no axis, no title, no legend), using palette colors
    mask_only_path = output_path.replace('.png', '_mask.png')
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.imshow(mask_img.convert("RGBA"))
    ax2.axis('off')
    fig2.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig2.savefig(mask_only_path, bbox_inches='tight', pad_inches=0, dpi=150)
    print(f"Saved mask-only segmentation to {mask_only_path}")
    plt.close(fig2)

if __name__ == "__main__":
    main()
