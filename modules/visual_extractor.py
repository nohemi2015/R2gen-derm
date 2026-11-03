import torch
import torch.nn as nn
import torchvision.models as models

# Feature dimension of each supported backbone (used to set --d_vf automatically).
FEATURE_DIMS = {
    'resnet101': 2048,
    'resnet50': 2048,
    'vit_base_patch16_224.augreg_in21k': 768,
    'vit_large_patch16_224.augreg_in21k': 1024,
}


def is_vit(name):
    return name.startswith('vit')


def feature_dim(name):
    if name not in FEATURE_DIMS:
        raise ValueError('Unknown visual extractor: {}'.format(name))
    return FEATURE_DIMS[name]


class VisualExtractor(nn.Module):
    """Returns (patch_feats [B, N, d_vf], avg_feats [B, d_vf]).

    resnet*: torchvision ImageNet-1k weights, 7x7=49 patches (original R2Gen).
    vit_*:   timm ImageNet-21k weights, 14x14=196 patch tokens (CLS token dropped).
    """

    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.vit = is_vit(self.visual_extractor)
        if self.vit:
            import timm
            self.model = timm.create_model(self.visual_extractor, pretrained=self.pretrained, num_classes=0)
        else:
            weights = 'DEFAULT' if self.pretrained else None
            model = getattr(models, self.visual_extractor)(weights=weights)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        if self.vit:
            tokens = self.model.forward_features(images)          # [B, prefix + N, D]
            patch_feats = tokens[:, self.model.num_prefix_tokens:]  # drop CLS
            avg_feats = patch_feats.mean(dim=1)
            return patch_feats, avg_feats
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
