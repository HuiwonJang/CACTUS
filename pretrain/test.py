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

    dataset = args.dataset
    dataset = datasets.get_dataset(dataset, args.datadir)
    loader  = datasets.get_loader(args, dataset)

    model = models.get_model(args, input_shape=dataset['input_shape'])
    model = idist.auto_model(model, sync_bn=True)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model_state = ckpt['model']
    model.load_state_dict(model_state)

    logger = setup_logger(name='logging')

    if args.eval_fewshot_metric == 'linear-eval':
        val  = 0, 0 # utils.evaluate_fewshot_linear(model, loader['val'])
        test = utils.evaluate_fewshot_linear(model, loader['test'])
    else:
        val  = 0, 0 # utils.evaluate_fewshot(model, loader['val'],  args.eval_fewshot_metric)
        test = utils.evaluate_fewshot(model, loader['test'], args.eval_fewshot_metric)

    logger.info(f'[Model: {args.model}] [dataset: {args.dataset}]'
                f'[{args.N} way {args.K} shot] [FewShot {args.eval_fewshot_metric}]'
                f'[{val[0]:.4f}±{val[1]:.4f}] | {test[0]:.4f}±{test[1]:.4f}]')


if __name__ == "__main__":
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--datadir', type=str, default='/data/miniimagenet')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--model', type=str, default='moco')
    parser.add_argument('--backbone', type=str, default='conv5')

    parser.add_argument('--prediction', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--queue-size', type=int, default=16384)
    parser.add_argument('--sinkhorn-iter', type=int, default=3)
    parser.add_argument('--n-prototypes', type=int, default=2048)

    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=15)
    parser.add_argument('--num-tasks', type=int, default=2000)

    parser.add_argument('--eval-fewshot-metric', type=str, default='knn')

    args = parser.parse_args()

    with idist.Parallel() as parallel:
        parallel.run(main, args)

