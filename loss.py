import torch
from torch import nn
from torch.nn import functional as F

def byol_loss(img_emb, text_emb):
    img_emb = F.normalize(img_emb, p=2, dim=-1)
    loss = 2 - 2 * (img_emb * text_emb).sum(dim=-1)
    return loss.mean()

def build_mlp(in_dim=512, hidden_dim=2048, out_dim=1024, bn=True, GELU=False):
    layers = [nn.Linear(in_dim, hidden_dim, bias=False if bn else True)]
    if bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    if GELU:
        layers.append(nn.GELU())
    else:
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)

# def concat_all_gather(tensor, rank=None, world_size=1):
#     """
#     rank=None means no gradient will be retained.
#     Specify rank with a int to retain gradient on local rank.
#     """
#     tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
#     torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
#     if rank is not None:
#         tensors_gather[rank] = tensor  # retain gradients on local rank
#     output = torch.cat(tensors_gather, dim=0)
#     return output

def ctrastive_loss(img_emb, text_emb, tau=0.05):
    # half of clip_loss ?
    img_emb = F.normalize(img_emb, p=2, dim=-1)
    # print(img_emb.shape, text_emb.shape)
    text_emb = torch.flatten(text_emb, start_dim=1, end_dim=-1)
    # print(img_emb.shape, text_emb.shape)
    logits = img_emb @ text_emb.T / tau  # temperature
    labels = torch.arange(logits.shape[0], dtype=torch.long, device=img_emb.device)
    return F.cross_entropy(logits, labels)

def clip_loss(img_emb, text_emb, tau=0.05):
    img_emb = F.normalize(img_emb, p=2, dim=-1)
    # all_image_features = concat_all_gather(img_emb)
    # all_text_features = concat_all_gather(text_emb)
    # logits = (all_image_features @ all_text_features.T) / tau
    logits = (img_emb @ text_emb.T) / tau
    labels = torch.arange(logits.shape[0], dtype=torch.long, device=img_emb.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.
    return loss