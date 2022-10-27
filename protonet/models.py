import torch
import torch.nn as nn
import torch.nn.functional as F


def get_backbone(backbone, input_shape):
    if backbone == 'conv4':
        layers = []
        in_channels = input_shape[0]
        for _ in range(4):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)))
            in_channels = 64
        layers.append(nn.Flatten())
        net = nn.Sequential(*layers)
        net.out_dim = 1600

    else:
        raise NotImplementedError

    return net


def get_model(args, input_shape):
    return ProtoNet(backbone=args.backbone,
                     input_shape=input_shape,
                     )


class ProtoNet(nn.Module):
    def __init__(self,
                 backbone: str = 'conv4',
                 input_shape: tuple[int] = (3, 84, 84),
                 ):
        super().__init__()
        self.backbone = get_backbone(backbone, input_shape)
        self.input_shape = input_shape

    def forward(self, batch, mode='train', **kargs):
        if mode == 'train':
            return self.compute_loss(batch, **kargs)
        elif mode == 'feature':
            return self.backbone(batch)

    def compute_loss(self, batch, N, K, Q):
        x, _ = batch
        x = x.view(N, K+Q, *self.input_shape)

        shots   = x[:, :K].reshape(N*K, *self.input_shape)
        queries = x[:, K:].reshape(N*Q, *self.input_shape)
        shots   = self.backbone(shots)
        queries = self.backbone(queries)

        prototypes = shots.view(N, K, shots.shape[-1]).mean(dim=1)

        sq_distances = torch.sum((prototypes.unsqueeze(0)-queries.unsqueeze(1))**2, dim=-1)
        targets = torch.arange(N, device=queries.device).repeat_interleave(Q)
        loss = F.cross_entropy(-sq_distances, targets)

        return dict(loss=loss)

