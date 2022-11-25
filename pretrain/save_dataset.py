import os
import argparse

import torch
import torch.backends.cudnn as cudnn

import torchvision.transforms as T

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor, setup_logger

import utils
import models
import datasets


def get_dataset(datadir):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    transform = T.Compose([T.ToTensor(),
                           T.Normalize(mean=mean, std=std)])

    train = datasets.MiniImageNet(os.path.join(datadir, 'train'), transform=transform)
    val   = datasets.MiniImageNet(os.path.join(datadir, 'val'),   transform=transform)
    test  = datasets.MiniImageNet(os.path.join(datadir, 'test'),  transform=transform)
    num_classes = (64, 16, 20)
    input_shape = (3, 84, 84)

    return dict(train=train,
                val=val,
                test=test,
                num_classes=num_classes,
                input_shape=input_shape)


def get_loader(args, dataset):
    loader = {}
    for mode in ['train', 'val', 'test']:
        loader[mode] = idist.auto_dataloader(dataset[mode],
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers,
                                             drop_last=False,
                                             pin_memory=True)

    return loader

def main(local_rank, args):
    device = idist.device()

    dataset = get_dataset(args.datadir)
    loader  = get_loader(args, dataset)

    model = models.get_model(args, input_shape=dataset['input_shape'])
    model = idist.auto_model(model, sync_bn=True)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model_state = ckpt['model']
    model.load_state_dict(model_state)

    logger = setup_logger(name='logging')

    utils.save_features(model, loader, args.savedir, args.model, args.backbone, args.n_clusters)


if __name__ == "__main__":
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--datadir', type=str, default='/data/miniimagenet')
    parser.add_argument('--savedir', type=str, default='./data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='moco')
    parser.add_argument('--backbone', type=str, default='conv5')
    parser.add_argument('--n-clusters', type=int, default=512)

    parser.add_argument('--prediction', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--queue-size', type=int, default=16384)
    parser.add_argument('--sinkhorn-iter', type=int, default=3)
    parser.add_argument('--n-prototypes', type=int, default=2048)

    args = parser.parse_args()

    with idist.Parallel() as parallel:
        parallel.run(main, args)

