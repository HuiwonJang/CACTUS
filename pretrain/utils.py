import os
import math
import copy
import faiss
import pickle

from sklearn.neighbors import KNeighborsClassifier

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
def collect_features(model, loader):
    model.eval()
    device = idist.device()
    Y, Z, paths = [], [], []
    for (i, batch) in enumerate(loader):
        x, y = convert_tensor(batch[0], device=device)
        full_path = batch[1]
        path = [path.split('/')[-2:] for path in full_path] #folder & file_name
        z = model(x, mode='feature')
        Y.append(y.detach())
        Z.append(z.detach())
        paths.append(path)
        print(f'{i+1:4d} / {len(loader):4d}', end='\r')
    paths = sum(paths, [])
    Y = torch.cat(Y).detach()
    Z = torch.cat(Z).detach()
    return Y, Z, paths


@torch.no_grad()
def save_features(model, loader, datadir, model_name, backbone_name, n_cluster, n_partitions=100):
    save_root = os.path.join(datadir, f'{model_name}_{backbone_name}_{n_cluster}')
    if idist.get_rank() == 0:
        os.makedirs(save_root)

    #train mode: save x, clst_ids (n_partition x N)
    _, Z, paths = collect_features(model, loader['train'])
    Z = Z.cpu().numpy()

    Y = []
    for partition in range(n_partitions):
        kmeans = faiss.Kmeans(d=Z.shape[1], k=n_cluster, niter=10)
        kmeans.train(Z)
        _, assignments = kmeans.index.search(Z, 1)
        Y.append(np.array(assignments.reshape(-1)))
        print(f'obtaining partition: {partition+1:4d} / {n_partitions:4d} done', end='\r')
    Y = np.stack(Y) #100xN

    save_dict = {'paths': paths,
                 'Y': Y}
    with open(os.path.join(save_root, 'train.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)

    #val, test mode: save x, y
    for mode in ['val', 'test']:
        Y, _, paths = collect_features(model, loader[mode])
        Y = Y.cpu().numpy()

        save_dict = {'paths': paths,
                     'Y': Y}
        with open(os.path.join(save_root, f'{mode}.pkl'), 'wb') as f:
            pickle.dump(save_dict, f)


@torch.no_grad()
def evaluate_fewshot(model, loader, metric):
    model.eval()
    device = idist.device()

    N = loader.batch_sampler.N
    K = loader.batch_sampler.K
    Q = loader.batch_sampler.Q
    accuracies = []
    for cnt, task in enumerate(loader):
        x, _ = convert_tensor(task[0], device=device)
        input_shape = x.shape[1:]
        x = x.view(N, K+Q, *input_shape)

        shots   = x[:, :K].reshape(N*K, *input_shape)
        queries = x[:, K:].reshape(N*Q, *input_shape)
        if metric == 'knn':
            knn = KNeighborsClassifier(n_neighbors=K, metric='cosine')
            shots_knn   = F.normalize(model(shots,   mode='feature')).detach().cpu().numpy()
            queries_knn = F.normalize(model(queries, mode='feature')).detach().cpu().numpy()

            y_shots = np.tile(np.expand_dims(np.arange(N), 1), K).reshape(-1)
            knn.fit(shots_knn, y_shots)

            preds = np.array(knn.predict(queries_knn))
            labels = np.tile(np.expand_dims(np.arange(N), 1), Q).reshape(-1)
            accuracies.append((preds == labels).mean().item())

        else:
            raise NotImplementedError

        print(f'{cnt:4d} {sum(accuracies)/len(accuracies):.4f}', end='\r')

    accuracies = idist.all_gather(torch.tensor(accuracies))
    return accuracies.mean(), accuracies.std()*1.96/math.sqrt(accuracies.numel())


def evaluate_fewshot_linear(model, loader):
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
        y_shots = torch.arange(N, device=shots.device).repeat_interleave(K)
        labels  = torch.arange(N, device=shots.device).repeat_interleave(Q)

        net        = copy.deepcopy(model.backbone)
        classifier = nn.Linear(net.out_dim, N).to(device)
        net.eval()
        classifier.train()

        optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            shots   = net(shots)
            queries = net(queries)

        for _ in range(100):
            with torch.no_grad():
                shots   = shots.detach()
                queries = queries.detach()

            rand_id = np.random.permutation(N*K)
            batch_indices = [rand_id[i*4:(i+1)*4] for i in range(rand_id.size//4)]
            for id in batch_indices:
                x_train = shots[id]
                y_train = y_shots[id]
                shots_pred = classifier(x_train)
                loss = criterion(shots_pred, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        net.eval()
        classifier.eval()
        with torch.no_grad():
            preds = classifier(queries).argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            accuracies.append(acc)
        print(f'{cnt:4d} {sum(accuracies)/len(accuracies):.4f}', end='\r')

    accuracies = idist.all_gather(torch.tensor(accuracies))
    return accuracies.mean(), accuracies.std()*1.96/math.sqrt(accuracies.numel())

