import argparse

import torch
import torch.backends.cudnn as cudnn

import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor, setup_logger

import utils
import models
import datasets


def main(local_rank, args):
    device = idist.device()

    dataset = datasets.get_dataset(args.dataset, args.datadir, args.pkldir)
    loader  = datasets.get_loader(args, dataset)

    model = models.get_model(args, input_shape=dataset['input_shape'])
    model = idist.auto_model(model, sync_bn=True)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    logger = setup_logger(name='logging')

    val  = 0, 0
    test = utils.evaluate_fewshot(model, loader['test'])

    logger.info(f'[FewShot] '
                f'[{val[0]:.4f}±{val[1]:.4f}] | {test[0]:.4f}±{test[1]:.4f}]')


if __name__ == "__main__":
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--pkldir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--datadir', type=str, default='/data/miniimagenet')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--backbone', type=str, default='conv4')

    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=15)
    parser.add_argument('--num-tasks', type=int, default=2000)

    args = parser.parse_args()

    with idist.Parallel() as parallel:
        parallel.run(main, args)

