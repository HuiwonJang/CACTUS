import os
import argparse

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

import ignite.distributed as idist
from ignite.engine import Engine, Events, State
from ignite.utils import convert_tensor

import utils
import models
import datasets


def main(local_rank, args):
    device = idist.device()
    logger, tb_logger = utils.get_logger(args)
    dataset = datasets.get_dataset(args.dataset, args.datadir, args.pkldir)
    loader  = datasets.get_loader(args, dataset)

    model = models.get_model(args, input_shape=dataset['input_shape'])
    model = idist.auto_model(model, sync_bn=True)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                              lr=args.lr, momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=args.lr)
    optimizer = idist.auto_optim(optimizer)

    if args.scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs*len(loader['train']))
    elif args.scheduler == 'fixed':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    def training_step(engine, batch):
        model.train()
        batch = convert_tensor(batch, device=device, non_blocking=True)
        outputs = model(batch, N=args.N, K=args.K, Q=args.Q)
        optimizer.zero_grad()
        outputs['loss'].backward()
        optimizer.step()
        scheduler.step()
        return outputs

    trainer = Engine(training_step)
    if logger is not None:
        trainer.logger = logger
        trainer.tb_logger = tb_logger
    trainer.add_event_handler(Events.ITERATION_COMPLETED, utils.log)

    @idist.one_rank_only()
    @trainer.on(Events.ITERATION_COMPLETED(every=args.eval_freq))
    def evaluation_step(engine):
        val  = utils.evaluate_fewshot(model, loader['val'])
        test = utils.evaluate_fewshot(model, loader['test'])

        if idist.get_rank() == 0:
            iter = engine.state.iteration
            engine.logger.info(f'[Iter {iter:4d}] '
                               f'[FewShot {val[0]:.4f}±{val[1]:.4f}] | {test[0]:.4f}±{test[1]:.4f}]')
            engine.tb_logger.add_scalar(f'fewshot/val',  val[0],  iter)
            engine.tb_logger.add_scalar(f'fewshot/test', test[0], iter)
        idist.barrier()

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=args.save_freq), utils.save_checkpoint, args,
                              model=model, optimizer=optimizer, scheduler=scheduler)

    trainer.run(loader['train'], max_epochs=args.num_epochs)
    if tb_logger is not None:
        tb_logger.close()


if __name__ == "__main__":
    cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--pkldir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='miniimagenet')
    parser.add_argument('--datadir', type=str, default='/data/miniimagenet')
    parser.add_argument('--num-epochs', type=int, default=60)
    parser.add_argument('--base-lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='step', choices=['cos', 'fixed', 'step'])

    parser.add_argument('--save-freq', type=int, default=10)
    parser.add_argument('--eval-freq', type=int, default=100)

    parser.add_argument('--backbone', type=str, default='conv4')

    # for evaluation
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--Q', type=int, default=15)
    parser.add_argument('--num-tasks', type=int, default=1000)

    args = parser.parse_args()
    args.lr = args.base_lr

    n = torch.cuda.device_count()
    if n == 1:
        with idist.Parallel() as parallel:
            parallel.run(main, args)
    else:
        with idist.Parallel(backend='nccl', nproc_per_node=n, master_port=os.environ.get('MASTER_PORT', 2222)) as parallel:
            parallel.run(main, args)

