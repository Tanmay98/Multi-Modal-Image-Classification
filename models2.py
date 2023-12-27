import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

__all__ = ['deit_small_patch16_224']


class VisionTransformer(TimmVisionTransformer):
    def __init__(self, *args, proj_type='linear', proj_dim=512, **kwargs):
        super(VisionTransformer, self).__init__(*args, **kwargs)

        if proj_type is None:
            print('no emb projection')
            self.proj_layer = nn.Identity()
        elif proj_type == 'linear':
            print('building linear projection')
            self.proj_layer = nn.Linear(self.embed_dim, proj_dim, bias=False)
            trunc_normal_(self.proj_layer.weight, std=.02)
        # elif proj_type == 'mlp':
        #     print('building mlp')
        #     self.proj_layer = build_mlp(in_dim=self.embed_dim, hidden_dim=int(2*proj_dim), out_dim=proj_dim, bn=False)
        #     trunc_normal_(self.proj_layer[0].weight, std=.02)
        #     trunc_normal_(self.proj_layer[-1].weight, std=.02)
        else:
            raise NotImplementedError

    def forward(self, x, get_feat=False):
        feat = self.forward_features(x)
        pred = self.head(feat)
        if get_feat:
            return self.proj_layer(feat), pred
        else:
            return pred

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model