import os
import json
import jsonlines
import random
from collections import defaultdict

import pandas as pd
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torch
import torch.nn as nn
import torchvision.datasets as D
import torchvision.transforms as T
import ignite.distributed as idist
from torchvision.datasets.utils import list_files



class RandomNoise(object):
    def __init__(self, ratio):
        self.ratio = ratio

    def __call__(self, x):
        noise = np.random.choice([-1, 0, 1], x.shape[0], p=[self.ratio/2, 1-self.ratio, self.ratio/2])
        x = np.abs(x-noise)
        return x


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MultipleTransform(nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, x):
        return [t(x) for t in self.transforms]


class FewShotTaskSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, N, K, Q, num_tasks):
        self.N = N
        self.K = K
        self.Q = Q
        self.num_tasks = num_tasks

        if isinstance(dataset, (D.ImageFolder)):
            labels = [y for _, y in dataset.samples]
        else:
            raise NotImplementedError

        self.indices = defaultdict(list)
        for i, y in enumerate(labels):
            self.indices[y].append(i)

    def __iter__(self):
        for _ in range(self.num_tasks):
            batch_indices = []
            labels = random.sample(list(self.indices.keys()), self.N)
            for y in labels:
                if len(self.indices[y]) >= self.K+self.Q:
                    batch_indices.extend(random.sample(self.indices[y], self.K+self.Q))
                else:
                    batch_indices.extend(random.choices(self.indices[y], k=self.K+self.Q))
            yield batch_indices


def get_augmentation(dataset, method='none'):
    interpolation=T.InterpolationMode.BICUBIC
    if dataset == 'miniimagenet':
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if method == 'none':
            return T.Compose([T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'strong':
            return T.Compose([T.RandomResizedCrop(84, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                              T.RandomGrayscale(p=0.2),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
        elif method == 'weak':
            return T.Compose([T.RandomResizedCrop(84, scale=(0.2, 1.), interpolation=interpolation),
                              T.RandomHorizontalFlip(),
                              T.ToTensor(),
                              T.Normalize(mean=mean, std=std)])
    else:
        raise NotImplementedError


def get_dataset(dataset, datadir, augmentations=['strong', 'strong']):
    if dataset == 'miniimagenet':
        augs = [get_augmentation(dataset, aug) for aug in augmentations]
        train = D.ImageFolder(os.path.join(datadir, 'train'), transform=MultipleTransform(augs))
        val   = D.ImageFolder(os.path.join(datadir, 'val'),  transform=get_augmentation(dataset, 'none'))
        test  = D.ImageFolder(os.path.join(datadir, 'test'), transform=get_augmentation(dataset, 'none'))
        num_classes = (64, 16, 20)
        input_shape = (3, 84, 84)
    else:
        raise Exception(f'Unknown Dataset: {dataset}')

    return dict(train=train,
                val=val,
                test=test,
                num_classes=num_classes,
                input_shape=input_shape)


def get_loader(args, dataset, splits=['train', 'val', 'test']):
    loader = {}
    loader['train'] = idist.auto_dataloader(dataset['train'],
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers,
                                            shuffle=True, drop_last=True,
                                            pin_memory=True)
    for split in ['val', 'test']:
        batch_sampler = FewShotTaskSampler(dataset[split], N=args.N, K=args.K, Q=args.Q,
                                           num_tasks=args.num_tasks // idist.get_world_size())
        loader[split] = torch.utils.data.DataLoader(dataset[split],
                                                    batch_sampler=batch_sampler,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)

    return loader

