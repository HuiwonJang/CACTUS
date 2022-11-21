import os
import json
import random
from collections import defaultdict

from PIL import Image

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


'''
dataset Example.
TODO: open file of (x_1, y_pseudo1), ..., (x_N, y_pseudoN)

class JSONImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, root, split, transform, depth=0):
        super().__init__()

        with open(f'splits/{dataset}/{split}.json', 'r') as f:
            dir_list = json.load(f)
        class_to_idx = {category: i for i, category in enumerate(dir_list['label_names'])}

        self.samples = []

        for path in dir_list['image_names']:
            file_path = os.path.join(root, path)
            category = path.split('/')[depth]
            self.samples.append((file_path, class_to_idx[category]))

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filename, y = self.samples[index]
        img = Image.open(filename, mode='r').convert("RGB")
        return self.transform(img), y
'''


def get_augmentation(dataset):
    if dataset == 'miniimagenet':
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        return T.Compose([T.ToTensor(),
                          T.Normalize(mean=mean, std=std)])
    else:
        raise NotImplementedError


def get_dataset(dataset, datadir):
    transform = get_augmentation(dataset)
    if dataset == 'miniimagenet':
        train = D.ImageFolder(os.path.join(datadir, 'train'), transform=transform)
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
    train_batch_sampler = FewShotTaskSampler(dataset['train'], N=args.N, K=args.K, Q=args.Q,
                                             num_tasks=args.num_tasks // idist.get_world_size())
    loader['train'] = torch.utils.data.DataLoader(dataset['train'],
                                                  batch_sampler=train_batch_sampler,
                                                  num_workers=args.num_workers,
                                                  pin_memory=True)

    for split in ['val', 'test']:
        batch_sampler = FewShotTaskSampler(dataset[split], N=5, K=args.K, Q=args.Q,
                                           num_tasks=args.num_tasks // idist.get_world_size())
        loader[split] = torch.utils.data.DataLoader(dataset[split],
                                                    batch_sampler=batch_sampler,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)

    return loader

