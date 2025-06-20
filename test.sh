# ORIGINAL, but demo_e200.ckpt instead of lseg_ade20k_l16.ckpt
# --eval argument: yes, then metrics; no, then images
export CUDA_VISIBLE_DEVICES=0; python test_lseg.py --backbone clip_vitl16_384 --dataset ade20k --data-path ../datasets/ \
--weights checkpoints/demo_e200.ckpt --widehead --no-scaleinv 


# MY CODE on LERF Teatime dataset
#export CUDA_VISIBLE_DEVICES=0; python test_lseg.py --backbone clip_vitl16_384 --eval --dataset lerf --data-path /home/neural_fields/Unified-Lift-Gabor/data/lerf/teatime/images_train \
#--weights checkpoints/demo_e200.ckpt --widehead --no-scaleinv 




