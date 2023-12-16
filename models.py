from timm.models.fastvit import FastVit as TimFastVit
from timm.models.fastvit import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from functools import partial
import torch
import torch.nn as nn
from loss import build_mlp

__all__ = ['fastvit_s12']

class FastVit(TimFastVit):
    def __init__(self, proj_dim=512, **kwargs):
        super().__init__(num_classes=10)

        self.proj_layer = nn.Linear(65536, proj_dim, bias=False)
        trunc_normal_(self.proj_layer.weight, std=.02)

    def forward(self, x):
        feat = self.forward_features(x)
        pred = self.head(feat)

        # return pred
        return self.proj_layer(feat.flatten(start_dim=1, end_dim=-1)), pred

@register_model
def fastvit_s12(pretrained=False, **kwargs):
    model = FastVit(
        layers=(2, 2, 6, 2),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(4, 4, 4, 4),
        token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")
    )
    my_cfg = _cfg()
    my_cfg["mean"] = (0.5,0.5,0.5)
    my_cfg["std"] = (0.5,0.5,0.5)
    
    # model.default_cfg = _cfg()
    model.default_cfg = my_cfg
    
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://huggingface.co/timm/fastvit_t12.apple_in1k",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])

    return model