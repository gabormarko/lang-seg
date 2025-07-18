import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from additional_utils.models import LSeg_MultiEvalModule
from modules.lseg_module import LSegModule
import torchvision.transforms as transforms
import argparse

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
    if out_label_flag and labels is not None:
        u_index = np.unique(npimg)
        for i, index in enumerate(u_index):
            if index >= len(labels):
                continue
            label = labels[index]
            cur_color = [new_palette[index * 3] / 255.0, new_palette[index * 3 + 1] / 255.0, new_palette[index * 3 + 2] / 255.0]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches

class PadToMultipleOf32:
    def __call__(self, img):
        # img: torch.Tensor, shape [C, H, W]
        h, w = img.shape[1:]
        pad_h = (32 - h % 32) if h % 32 != 0 else 0
        pad_w = (32 - w % 32) if w % 32 != 0 else 0
        # Pad (left, top, right, bottom): here pad only on right and bottom
        return torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), value=0)

def main():
    parser = argparse.ArgumentParser(description="Batch LSeg inference for a folder of images")
    parser.add_argument('--input_dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--labels', type=str, nargs='+', default=None, help='List of class labels (space separated)')
    parser.add_argument('--weights', type=str, default='checkpoints/demo_e200.ckpt', help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='clip_vitl16_384', help='Model backbone')
    parser.add_argument('--ignore_index', type=int, default=255)
    parser.add_argument('--extract_features', action='store_true', help='If set, extract and save per-pixel feature embeddings instead of segmentation logits')
    args = parser.parse_args()

    # Default ADE20K labels if not provided
    if args.labels is None:
        labels = [
            'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass',
            'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair',
            # ... (add all ADE20K classes as needed)
        ]
    else:
        labels = args.labels
    new_palette = get_new_pallete(len(labels))

    torch.manual_seed(1)
    # Provide dummy data_path and dataset for model init
    # Inspect checkpoint for feature dimension info
    checkpoint = torch.load(args.weights, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    head1_keys = [k for k in state_dict.keys() if "head1.weight" in k]
    if head1_keys:
        head1_shape = state_dict[head1_keys[0]].shape
        print(f"[INFO] Checkpoint head1.weight shape: {head1_shape} (out_c, in_c, 1, 1)")
        print(f"[INFO] This means the checkpoint was trained with output feature dim: {head1_shape[0]}")
    else:
        print("[WARNING] Could not find head1.weight in checkpoint. Cannot determine feature dim.")

    module = LSegModule.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path="/tmp",  # dummy path
        dataset="lerf",    # dummy dataset name
        backbone=args.backbone,
        aux=False,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
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
    from encoding.models.sseg import BaseNet
    if isinstance(module.net, BaseNet):
        model = module.net
    else:
        model = module
    model = model.eval()
    model = model.cpu()
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]
    evaluator = LSeg_MultiEvalModule(
        model, scales=scales, flip=True
    ).cuda()
    evaluator.eval()

    # Input/output dirs
    input_dir = args.input_dir
    output_dir = args.output_dir
    seg_dir = os.path.join(output_dir, 'seg')
    comparison_dir = os.path.join(output_dir, 'comparisons')
    features_dir = os.path.join(output_dir, 'features')
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    # Try to load mean/std from mean_std.txt in input_dir    
    #mean_std_path = os.path.join(input_dir, 'mean_std.txt')
    #print(f"[DEBUG] Looking for mean_std.txt at: {mean_std_path}")
    #if os.path.exists(mean_std_path):
    #    arr = np.loadtxt(mean_std_path)
    #    mean = arr[0]
    #    std = arr[1]
    #    print(f"[INFO] Using dataset mean: {mean}, std: {std} from {mean_std_path}")
    #    mean = mean.tolist()
    #    std = std.tolist()
    #else:
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    print(f"[INFO] Using default mean/std: {mean}, {std}")

    # Harmonized preprocessing: fixed size resize to [360, 480], no aspect ratio preservation, no padding
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Resize([320, 480]),  # fixed size to match lseg_app.py
    ])

    # Always iterate over images in input_dir (no dataloader)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    dataset = [(transform(Image.open(os.path.join(input_dir, fname)).convert('RGB')), os.path.join(input_dir, fname)) for fname in image_files]

    for idx, (image, fname) in enumerate(tqdm(dataset)):
        base_name = os.path.splitext(os.path.basename(fname))[0]
        orig_pil = Image.open(fname).convert('RGB')  # Always reload original for true color
        print(f"[DEBUG] Processing {fname}")
        print(f"[DEBUG] Original image size: {orig_pil.size}")
        print(f"[DEBUG] Input image tensor shape: {image.shape}")
        pimage = image.unsqueeze(0) if isinstance(image, torch.Tensor) else transform(np.array(image)).unsqueeze(0)
        print(f"[DEBUG] Model input batch shape: {pimage.shape}")
        print(f"[DEBUG] Label list used for extraction: {labels}")
        # Save the preprocessed image used for feature extraction
        # Undo normalization for visualization
        preproc_img = image.clone()
        for t, m, s in zip(preproc_img, mean, std):
            t.mul_(s).add_(m)
        preproc_img = (preproc_img * 255).clamp(0, 255).byte().permute(1,2,0).cpu().numpy()
        print(f"[DEBUG] Preprocessed image shape: {preproc_img.shape}")
        preproc_pil = Image.fromarray(preproc_img)
        preproc_pil_path = os.path.join(features_dir, base_name + '_preproc.png')
        preproc_pil.save(preproc_pil_path)
        print(f"[DEBUG] Saved preprocessed image to {preproc_pil_path}")
        with torch.no_grad():
            if args.extract_features:
                # Extract per-pixel features
                device = next(model.net.parameters()).device
                pimage = pimage.to(device)
                features = model.net.extract_features(pimage)
                print(f"[DEBUG] Extracted features shape: {features.shape}")
                print(f"[DEBUG] Features file path: {os.path.join(features_dir, base_name + '.JPG.npy')}")
                if features.shape[1] != 512:
                    print(f"[WARNING] Feature dimension is {features.shape[1]}, expected 256. Check model weights and num_features setting.")
                else:
                    print(f"[INFO] Feature dimension confirmed as 512.")
                features_np = features.squeeze(0).cpu().numpy().astype(np.float16)  # [C, H, W] as float16
                npy_path = os.path.join(features_dir, base_name + '.JPG.npy')
                np.save(npy_path, features_np)
                print(f"[DEBUG] Saved features as float16 to {npy_path}")
            else:
                outputs = evaluator.parallel_forward(pimage, labels)
                print(f"[DEBUG] Model outputs: {[o.shape for o in outputs]}")
                if len(outputs) > 0:
                    print(f"[DEBUG] Output feature vector dimension: {outputs[0].shape[1]}")
                predicts = [torch.max(output, 1)[1].cpu().numpy() for output in outputs]
                pred = predicts[0]
                print(f"[DEBUG] Prediction mask shape: {pred.shape}")
                mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
                seg = mask.convert("RGBA")
                # Resize mask and seg to match original image size
                orig_size = orig_pil.size
                print(f"[DEBUG] Original image size: {orig_size}")
                mask_resized = mask.resize(orig_size, resample=Image.NEAREST)
                seg_resized = mask_resized.convert("RGBA")
                # Save mask in 'seg' subfolder
                mask_name = base_name + '_seg.png'
                mask_resized.save(os.path.join(seg_dir, mask_name))
                # Create overlay
                image_rgba = orig_pil.convert("RGBA")
                overlay = Image.blend(image_rgba, seg_resized, alpha=0.5)
                # Save overlay with legend using matplotlib
                fig_overlay, (ax_img, ax_legend) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [4, 1]})
                ax_img.imshow(overlay)
                ax_img.set_title("Overlay (Input + Segmentation)")
                ax_img.axis('off')
                if patches:
                    leg = ax_legend.legend(handles=patches, loc='center left', fontsize=14, frameon=True, borderaxespad=0.5)
                    ax_legend.set_axis_off()
                else:
                    ax_legend.text(0.5, 0.5, 'No legend available', ha='center', va='center', fontsize=16)
                    ax_legend.set_axis_off()
                plt.tight_layout()
                overlay_path = os.path.join(comparison_dir, base_name + '_overlay.png')
                fig_overlay.savefig(overlay_path, bbox_inches='tight', dpi=150)
                plt.close(fig_overlay)

                # --- Save side-by-side comparison: original + mask ---
                fig_compare, (ax_orig, ax_mask) = plt.subplots(1, 2, figsize=(16, 8))
                ax_orig.imshow(orig_pil)
                ax_orig.set_title('Original Image', fontsize=18)
                ax_orig.axis('off')
                ax_mask.imshow(mask_resized)
                ax_mask.set_title('Segmentation Mask', fontsize=18)
                ax_mask.axis('off')
                # Add legend to mask if available
                if patches:
                    fig_compare.legend(handles=patches, loc='lower center', fontsize=14, ncol=4, bbox_to_anchor=(0.5, -0.05))
                plt.tight_layout()
                comparison_path = os.path.join(comparison_dir, base_name + '_compare.png')
                fig_compare.savefig(comparison_path, bbox_inches='tight', dpi=150)
                plt.close(fig_compare)
    print(f"Batch inference complete. Processed {len(dataset)} images.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during batch inference: {e}")
