import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.backbone import res50, bbn_res50, res32_cifar, bbn_res32_cifar
from lib.models import EfficientNet, load_state_dict_from_url, model_urls, LOCAL_PRETRAINED
from lib.modules import GAP, Identity, FCNorm


def Efficientnet(model_name, pretrained, cfg, test=False):
    '''
    model_name :'efficientnet-b0', 'efficientnet-b1-7'
    '''
    model = EfficientNet.from_name(model_name)
    if not test:
        # if pretrained == False:
        #     state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        # else:

        # state_dict = torch.load(LOCAL_PRETRAINED[model_name])
        model_dict = model.state_dict()
        pretrained_dict = torch.load(cfg.BACKBONE.PRETRAINED_MODEL)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


    # fc_features = model._fc.in_features
    # model._fc = nn.Linear(fc_features, num_classes)
    return model


class Network(nn.Module):
    def __init__(self, cfg, mode="train", num_classes=1000):
        super(Network, self).__init__()
        pretrain = (
            True
            if mode == "train"
               and cfg.RESUME_MODEL == ""
               and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.num_classes = num_classes
        self.cfg = cfg
        if 'efficientnet' in self.cfg.BACKBONE.TYPE:
            self.backbone = Efficientnet(self.cfg.BACKBONE.TYPE, pretrain, self.cfg, test=False)
        else:
            self.backbone = eval(self.cfg.BACKBONE.TYPE)(
                self.cfg,
                pretrain=pretrain,
                pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
                last_layer_stride=2,
            )
        self.module = self._get_module()
        self.classifier = self._get_classifer()
        self.feature_len = self.get_feature_length()

    def forward(self, x, **kwargs):
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "classifier_flag" in kwargs:
            return self.classifier(x)

        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)

        x = self.classifier(x)
        return x

    def extract_feature(self, x, **kwargs):
        if "bbn" in self.cfg.BACKBONE.TYPE:
            x = self.backbone(x, **kwargs)
        else:
            x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)

        return x

    def freeze_backbone(self):
        print("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        print("Backbone has been loaded...")

    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v
        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Model has been loaded...")

    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        else:
            num_features = 2048

        if "bbn" in self.cfg.BACKBONE.TYPE:
            num_features = num_features * 2
        return num_features

    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module = Identity()
        else:
            raise NotImplementedError

        return module

    def _get_classifer(self):
        bias_flag = self.cfg.CLASSIFIER.BIAS

        num_features = self.get_feature_length()
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE == "FC":
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        else:
            raise NotImplementedError

        return classifier
