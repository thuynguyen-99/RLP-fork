import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from rlp.utils import load_img, random_add_jpg_compression
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


def is_image_file(filename):
    return any(
        filename.endswith(ext)
        for ext in ["jpeg", "JPEG", "jpg", "png", "JPG", "PNG", "gif"]
    )


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if img.size == 0:
        return img
    m = float(img.max())
    if m <= 1.0 + 1e-6:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def get_train_transform(ps: int):
    return A.Compose(
        [
            A.RandomResizedCrop(ps, ps, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.8),
            A.Gamma((90, 115), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=4, sat_shift_limit=8, val_shift_limit=8, p=0.3
            ),
            ToTensorV2(),
        ]
    )


def get_eval_transform():
    return A.Compose([ToTensorV2()])


class DatasetTrain(Dataset):
    def __init__(
        self, data_dir, input_folder="rainy", gt_folder="gt", img_options=None
    ):
        super().__init__()

        input_filenames = sorted(os.listdir(os.path.join(data_dir, input_folder)))
        gt_filenames = sorted(os.listdir(os.path.join(data_dir, gt_folder)))

        self.input_paths = [
            os.path.join(data_dir, input_folder, x)
            for x in input_filenames[:500]
            if is_image_file(x)
        ]
        self.gt_paths = [
            os.path.join(data_dir, gt_folder, x)
            for x in gt_filenames[:500]
            if is_image_file(x)
        ]

        self.img_options = img_options or {}
        self.ps = int(self.img_options.get("patch_size", 256))

        self._train_tf = get_train_transform(self.ps)
        self._eval_tf = get_eval_transform()

        self.img_num = len(self.input_paths)

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        tar_index = index % self.img_num

        inp_np = np.float32(load_img(self.input_paths[tar_index]))
        gt_np = np.float32(load_img(self.gt_paths[tar_index]))

        inp_np = _to_uint8(inp_np)
        gt_np = _to_uint8(gt_np)

        input_name = os.path.split(self.input_paths[tar_index])[-1]
        gt_name = os.path.split(self.gt_paths[tar_index])[-1]

        data = self._train_tf(image=inp_np, mask=gt_np)
        input_t = data["image"]
        gt_t = data["mask"]

        return input_t, gt_t, input_name, gt_name


class DatasetTest(Dataset):
    def __init__(self, inp_dir):
        super().__init__()
        inp_files = sorted(os.listdir(inp_dir))
        self.inp_paths = [
            os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)
        ]
        self.inp_num = len(self.inp_paths)

    def __len__(self):
        return self.inp_num

    def __getitem__(self, index):
        inp_path = self.inp_paths[index]
        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]
        inp = Image.open(inp_path)
        inp = TF.to_tensor(inp)
        return inp, filename
