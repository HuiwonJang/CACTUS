import os
import math
import copy
import torch
from torch.utils.tensorboard import SummaryWriter
from ignite.utils import setup_logger, convert_tensor
import ignite.distributed as idist


def get_logger(args):
    if idist.get_rank() == 0:
        os.makedirs(args.logdir)
        logger = setup_logger(name='logging', filepath=os.path.join(args.logdir, 'log.txt'))
        logger.info(args)
        logger.info(' '.join(os.sys.argv))
        tb_logger = SummaryWriter(log_dir=args.logdir)
    else:
        logger, tb_logger = None, None

    idist.barrier()
    return logger, tb_logger

@idist.one_rank_only()
def log(engine):
    if engine.state.iteration % 10 == 0:
        engine.logger.info(f'[Epoch {engine.state.epoch:4d}] '
                           f'[Iter {engine.state.iteration:6d}] '
                           f'[Loss {engine.state.output["loss"].item():.4f}]')
        for k, v in engine.state.output.items():
            engine.tb_logger.add_scalar(k, v, engine.state.iteration)


@idist.one_rank_only()
def save_checkpoint(engine, args, **kwargs):
    state = { k: v.state_dict() for k, v in kwargs.items() }
    state['engine'] = engine.state_dict()
    # torch.save(state, os.path.join(args.logdir, f'ckpt-{engine.state.epoch}.pth'))
    torch.save(state, os.path.join(args.logdir, f'last.pth'))


@torch.no_grad()
def evaluate_fewshot(model, loader):
    model.eval()
    device = idist.device()

    N = loader.batch_sampler.N
    K = loader.batch_sampler.K
    Q = loader.batch_sampler.Q
    accuracies = []
    for cnt, task in enumerate(loader):
        x, _ = convert_tensor(task, device=device)
        input_shape = x.shape[1:]
        x = x.view(N, K+Q, *input_shape)

        shots   = x[:, :K].reshape(N*K, *input_shape)
        queries = x[:, K:].reshape(N*Q, *input_shape)

        shots_  = model(shots,   mode='feature')
        queries = model(queries, mode='feature')

        prototypes = shots_.view(N, K, shots_.shape[-1]).mean(dim=1)

        sq_distances = torch.sum((prototypes.unsqueeze(0)-queries.unsqueeze(1))**2, dim=-1)
        _, preds = torch.min(sq_distances, dim=-1)
        labels = torch.arange(N, device=preds.device).repeat_interleave(Q)
        accuracies.append((preds == labels).float().mean().item())

        print(f'{cnt:4d} {sum(accuracies)/len(accuracies):.4f}', end='\r')

    accuracies = idist.all_gather(torch.tensor(accuracies))
    return accuracies.mean(), accuracies.std()*1.96/math.sqrt(accuracies.numel())