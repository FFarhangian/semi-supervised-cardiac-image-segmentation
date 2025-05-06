from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint

# Ignore warnings
import warnings

import pdb

warnings.filterwarnings("ignore")


def make_dataset(root, mode,supervision):
    assert mode in ['train','val', 'test']
    items = []
    if mode == 'train':
        if supervision == "full":
            train_img_path = os.path.join(root, 'train', 'Img')
            train_mask_path = os.path.join(root, 'train', 'GT')

            images = os.listdir(train_img_path)
            labels = os.listdir(train_mask_path)

            images.sort()
            labels.sort()

            for it_im, it_gt in zip(images, labels):
                item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
                items.append(item)
        elif supervision == 'semi':
            train_img_path = os.path.join(root, 'train', 'Img')
            train_mask_path = os.path.join(root, 'train', 'GT')
            unlabeled_img_path = os.path.join(root, 'train', 'Img-Unlabeled')

            images = os.listdir(train_img_path)
            unlabeled_images = os.listdir(unlabeled_img_path)
            images.extend(unlabeled_images)
            labels = os.listdir(train_mask_path)

            # images.sort()
            # labels.sort()
            labels.extend([None]*len(unlabeled_images))
            for it_im, it_gt in zip(images, labels):
                if it_gt:
                    item = (os.path.join(train_img_path, it_im), os.path.join(train_mask_path, it_gt))
                else:
                    item = (os.path.join(unlabeled_img_path, it_im), None)
                items.append(item)


    elif mode == 'val':
        val_img_path = os.path.join(root, 'val', 'Img')
        val_mask_path = os.path.join(root, 'val', 'GT')

        images = os.listdir(val_img_path)
        labels = os.listdir(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(val_img_path, it_im), os.path.join(val_mask_path, it_gt))
            items.append(item)
    else:
        test_img_path = os.path.join(root, 'test', 'Img')
        test_mask_path = os.path.join(root, 'test', 'GT')

        images = os.listdir(test_img_path)
        labels = os.listdir(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (os.path.join(test_img_path, it_im), os.path.join(test_mask_path, it_gt))
            items.append(item)

    return items


class MedicalImageDataset(Dataset):

    def __init__(self, mode, supervision, root_dir, transform=None, mask_transform=None, augment=False, equalize=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode,supervision)
        self.augmentation = augment
        self.equalize = equalize
        self.mode = mode
        self.supervision = supervision

    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            angle = random() * 60 - 30
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path)
        try:
            mask = Image.open(mask_path).convert('L')
        except:
            mask = np.zeros(np.array(img).shape)

        if self.equalize:
            img = ImageOps.equalize(img)

        if self.augmentation:
            img, mask = self.augment(img, mask)

        if self.transform:
            img = self.transform(img)
            mask = self.mask_transform(mask)
      
        return [img, mask, img_path]