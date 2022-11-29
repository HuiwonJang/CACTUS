import os
import pickle
import random
from collections import defaultdict

from PIL import Image

import numpy as np

import torch
import torchvision.datasets as D
import torchvision.transforms as T
import ignite.distributed as idist


class FewShotTaskSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, N, K, Q, num_tasks):
        self.N = N
        self.K = K
        self.Q = Q
        self.num_tasks = num_tasks

        if isinstance(dataset, D.ImageFolder):
            labels = [y for _, y in dataset.samples]
        else:
            raise NotImplementedError

        self.indices = defaultdict(list)
        for i, y in enumerate(labels):
            self.indices[y].append(i)

    def __len__(self):
        return self.num_tasks

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


class PartitionFewShotTaskSampler(torch.utils.data.BatchSampler):
    def __init__(self, dataset, N, K, Q, num_tasks):
        self.N = N
        self.K = K
        self.Q = Q
        self.num_tasks = num_tasks
        self.partition = dataset.partition

        if isinstance(dataset, PartitionDataset):
            labels = dataset.indices
        else:
            raise NotImplementedError

        self.indices = []
        for ys in labels:
            indices = defaultdict(list)
            for j, y in enumerate(ys):
                indices[y].append(j)
            self.indices.append(indices)

    def __len__(self):
        return self.num_tasks

    def __iter__(self):
        for _ in range(self.num_tasks):
            partition = np.random.randint(0, self.partition)
            indices = self.indices[partition]
            batch_indices = []
            labels = random.sample(list(indices.keys()), self.N)
            for y in labels:
                if len(indices[y]) >= self.K+self.Q:
                    batch_indices.extend(random.sample(indices[y], self.K+self.Q))
                else:
                    batch_indices.extend(random.choices(indices[y], k=self.K+self.Q))
            batch_indices = [idx+(self.partition*1000)*partition for idx in batch_indices]
            yield batch_indices


class PartitionDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, pkl_dir, transform):
        # train: paths: Nx2 (folder, file_name) || Y: 100xN (indices)
        with open(pkl_dir, 'rb') as f:
            data_list = pickle.load(f)
        data_paths = data_list['paths']

        self.transform = transform
        self.indices = data_list['Y']
        self.partition = self.indices.shape[0]

        self.data = []
        for paths in data_paths:
            self.data.append(os.path.join(data_root, *paths))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        partition = index//(self.partition*1000)
        idx = index%(self.partition*1000)
        filename, y = self.data[idx], self.indices[partition, idx]
        img = Image.open(filename, mode='r').convert("RGB")

        return self.transform(img), y


def get_augmentation(dataset):
    if dataset == 'miniimagenet':
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        return T.Compose([T.ToTensor(),
                          T.Normalize(mean=mean, std=std)])
    else:
        raise NotImplementedError


def get_dataset(dataset, datadir, pkldir):
    transform = get_augmentation(dataset)
    if dataset == 'miniimagenet':
        train = PartitionDataset(os.path.join(datadir, 'train'), os.path.join(pkldir, 'train.pkl'), transform=transform)
        #train = D.ImageFolder(os.path.join(datadir, 'train'), transform=transform)
        val   = D.ImageFolder(os.path.join(datadir, 'val'),   transform=transform)
        test  = D.ImageFolder(os.path.join(datadir, 'test'),  transform=transform)
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
    train_batch_sampler = PartitionFewShotTaskSampler(dataset['train'], N=args.N, K=args.K, Q=args.Q,#PartitionFewShotTaskSampler(dataset['train'], N=args.N, K=args.K, Q=args.Q,
                                                      num_tasks=args.num_tasks // idist.get_world_size())
    loader['train'] = torch.utils.data.DataLoader(dataset['train'],
                                                  batch_sampler=train_batch_sampler,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)

    for split in ['val', 'test']:
        batch_sampler = FewShotTaskSampler(dataset[split], N=5, K=args.K, Q=15,
                                           num_tasks=args.num_tasks // idist.get_world_size())
        loader[split] = torch.utils.data.DataLoader(dataset[split],
                                                    batch_sampler=batch_sampler,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)

    return loader

