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
    # Debug: print unique values and max value in the mask
    print('DEBUG: unique mask values:', np.unique(npimg))
    print('DEBUG: max mask value:', np.max(npimg))
    # Ensure palette is a flat list of ints and pad/crop to 768
    new_palette = list(map(int, new_palette))
    if len(new_palette) < 768:
        new_palette = new_palette + [0] * (768 - len(new_palette))
    else:
        new_palette = new_palette[:768]
    assert len(new_palette) == 768, f'Palette length is {len(new_palette)}, should be 768'
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

def main():
    parser = argparse.ArgumentParser(description="Batch LSeg inference for a folder of images")
    parser.add_argument('--input_dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--labels', type=str, nargs='+', default=None, help='List of class labels (space separated)')
    parser.add_argument('--weights', type=str, default='checkpoints/demo_e200.ckpt', help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='clip_vitl16_384', help='Model backbone')
    parser.add_argument('--ignore_index', type=int, default=255)
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Resize([360,480]),
    ])
    # Input/output dirs
    input_dir = args.input_dir
    output_dir = args.output_dir
    seg_dir = os.path.join(output_dir, 'seg')
    comparison_dir = os.path.join(output_dir, 'comparisons')
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)

    # Always iterate over images in input_dir (no dataloader)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    dataset = [(transform(Image.open(os.path.join(input_dir, fname)).convert('RGB')), os.path.join(input_dir, fname)) for fname in image_files]

    for idx, (image, fname) in enumerate(tqdm(dataset)):
        base_name = os.path.splitext(os.path.basename(fname))[0]
        pimage = image.unsqueeze(0) if isinstance(image, torch.Tensor) else transform(np.array(image)).unsqueeze(0)
        with torch.no_grad():
            outputs = evaluator.parallel_forward(pimage, labels)
            predicts = [torch.max(output, 1)[1].cpu().numpy() for output in outputs]
        pred = predicts[0]
        mask, patches = get_new_mask_pallete(pred, new_palette, out_label_flag=True, labels=labels)
        seg = mask.convert("RGBA")
        # Resize mask and seg to match original image size
        orig_pil = Image.open(fname).convert('RGB')  # Always reload original for true color
        orig_size = orig_pil.size
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
