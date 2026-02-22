import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import torchvision.transforms.functional as TF


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = np.array(img)
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = PIL.Image.open(path).convert("L")
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg[seg > 0] = 255
    seg = np.stack([seg == 0, seg == 255])
    seg = seg.astype(np.float32)
    return seg

def load_and_random_crop(image_path: pathlib.Path, mask_path: pathlib.Path, crop_size=512):
    """
    Load an image and its mask, apply the same random crop.

    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the corresponding mask.
        crop_size (int): Size of the square crop (default: 512).

    Returns:
        (PIL.Image.Image, PIL.Image.Image): Cropped image and mask
    """
    # Open image and mask
    image = Image.open(image_path)
    mask = Image.open(mask_path).convert("L")  # Assume mask is grayscale

    # Ensure both image and mask are the same size
    assert image.size == mask.size, "Image and mask must be the same size"

    w, h = image.size
    if w < crop_size or h < crop_size:
        raise ValueError(f"Image size {w}x{h} is smaller than crop size {crop_size}x{crop_size}")

    # Random top-left corner
    left = random.randint(0, w - crop_size)
    top = random.randint(0, h - crop_size)

    # Crop both image and mask
    image_cropped = TF.crop(image, top, left, crop_size, crop_size)
    mask_cropped = TF.crop(mask, top, left, crop_size, crop_size)

    img = np.array(image_cropped)
    seg = np.array(mask_cropped)

    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))

    seg[seg > 0] = 255
    seg = np.stack([seg == 0, seg == 255])
    seg = seg.astype(np.float32)

    return img, seg


@dataclass
class WBC(Dataset):

    subdataset: Literal["WorldRoad"]
    split: Literal["train", "val", "test"]
    label: Optional[Literal["road", "background"]] = None

    # train_txt_path: str = "/mnt/data1/pcx/hms/Segment_Zoo/best_folds/train_full_fold_huawei_sh_1.txt"
    # valid_txt_path: str = "/mnt/data1/pcx/hms/Segment_Zoo/best_folds/valid_full_fold_1.txt"
    # test_txt_path: str =  "/mnt/data1/pcx/hms/Segment_Zoo/best_folds/valid_full_fold_1.txt"

    train_txt_path: str = "/mnt/data1/pcx/hms/Segment_Zoo/data/ext_data/WorldRoadCrop_v1.0/newtrain.txt"
    valid_txt_path: str = "/mnt/data1/pcx/hms/Segment_Zoo/data/miniroad/test/test.txt"
    test_txt_path: str =  "/mnt/data1/pcx/hms/Segment_Zoo/data/miniroad/test/test.txt"

    # train_txt_path: str = "/mnt/data1/pcx/hms/Segment_Zoo/data/Global-Scale/train.txt"
    # valid_txt_path: str = "/mnt/data1/pcx/hms/Segment_Zoo/data/miniroad/test/test.txt"
    # test_txt_path: str =  "/mnt/data1/pcx/hms/Segment_Zoo/data/miniroad/test/test.txt"

    def __post_init__(self):

        if self.split == "train":
            self.txt_file = self.train_txt_path
        elif self.split == "val":
            self.txt_file = self.valid_txt_path
        else:
            self.txt_file = self.test_txt_path

        with open(self.txt_file, "r") as f:
            self.img_paths = [line.strip() for line in f.readlines()]

        if self.label is not None:
            self._ilabel = {"road": 1, "background": 0}[self.label]

    def _get_mask_path(self, img_path: str):

        if "/img/img" in img_path:
            return img_path.replace("/img/img", "/label/label")
        elif "sat" in img_path:
            return img_path.replace("sat", "gt")
        elif "/images/" in img_path:
            return img_path.replace("/images/", "/masks/")
        else:
            return img_path.replace("/img/", "/label/")

    def __len__(self):

        return len(self.img_paths)

    def __getitem__(self, idx):

        img_path = self.img_paths[idx]
        mask_path = self._get_mask_path(img_path)

        img = process_img(pathlib.Path(img_path), size=(512, 512))
        seg = process_seg(pathlib.Path(mask_path), size=(512, 512))

        # img, seg = load_and_random_crop(pathlib.Path(img_path), pathlib.Path(mask_path), 512)

        img_tensor = torch.from_numpy(img / 255.0)
        seg_tensor = torch.from_numpy(seg[self._ilabel][None])

        return img_tensor, seg_tensor

    @property
    def attr(self):
        return {
            "dataset": "WBC",
            "subdataset": self.subdataset,
            "modality": "Microscopy",
            "axis": 0,
            "label": self.label,
            "split": self.split,
        }
