# Fine-tuning Guide for RLP on SPA and Rain13k Datasets

This guide provides detailed instructions for fine-tuning the RLP (Rain Location Prior) model on SPA and Rain13k datasets.

## Configuration
```bash
TRAIN_PATH="/path/to/train/folder"
TRAIN_INPUT_FOLDER="rain"
TRAIN_GT_FOLDER="norain"
N_EPOCH=100
PRETRAINED_CHECKPOINT_PATH="/path/to/pretrained/checkpoint.pth"
```

## Dataset Structure
```
/kaggle/input/spadata/spa/train/
├── rain/          # Rainy images
└── norain/        # Clean images
```

## Fine-tuning Command
```bash
python train.py --arch Uformer_T \
                --batch_size 4 \
                --gpu 0 \
                --train_ps 256 \
                --train_dir $TRAIN_PATH \
                --save_dir ./logs \
                --dataset dataset_name \
                --warmup \
                --use_rlp \
                --use_rpim \
                --train_input_folder $TRAIN_INPUT_FOLDER \
                --train_gt_folder $TRAIN_GT_FOLDER \
                --nepoch $N_EPOCH \
                --pretrain_weights $PRETRAINED_CHECKPOINT_PATH
```

## Pre-trained Models
Download from [releases](https://github.com/zkawfanx/RLP/releases):
- [Uformer_T_RLP_RPIM.pth](https://github.com/zkawfanx/RLP/releases/download/v1.0.0/Uformer_T_RLP_RPIM.pth)
