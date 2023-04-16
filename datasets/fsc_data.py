import os
import random
from glob import glob

import torch.utils.data as data
import torchvision.transforms.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


class FSCData(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, is_gray=False,
                 method='train'):

        self.root_path = root_path
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))

        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0

        if is_gray:
            self.trans_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans_img = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.trans_dmap = transforms.ToTensor()

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('.jpg', '_dmap.npy')
        name = os.path.basename(img_path).split('.')[0]

        try:
            img = Image.open(img_path).convert('RGB')
            dmap = np.load(gd_path)
            dmap = dmap.astype(np.float32, copy=False)  # np.float64 -> np.float32 to save memory
        except:
            raise Exception('Image open error {}'.format(name))

        if self.method == 'train':
            return self.train_transform(img, dmap)
        elif self.method == 'val':
            return self.trans_img(img), np.sum(dmap), name

    def train_transform(self, img, dmap):
        dmap = Image.fromarray(dmap)
        wd, ht = img.size
        # random gray scale augmentation
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')

        # rescale augmentation
        re_size = random.random() * 0.5 + 0.75
        wdd = int(wd*re_size)
        htt = int(ht*re_size)
        if min(wdd, htt) >= self.c_size:
            raw_size = (wd, ht)
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            dmap = dmap.resize((wd, ht))
            ratio = (raw_size[0]*raw_size[1])/(wd*ht)
            dmap = Image.fromarray(np.array(dmap) * ratio)

        # random crop augmentation
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        dmap = F.crop(dmap, i, j, h, w)

        # random horizontal flip
        if random.random() > 0.5:
            img = F.hflip(img)
            dmap = F.hflip(dmap)

        return self.trans_img(img), self.trans_dmap(dmap)
