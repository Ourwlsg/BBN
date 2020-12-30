import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from lib.models.vision.resnet import Bottleneck, conv1x1

model_urls = {
    "resnet18": "https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth",
    "resnet34": "https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth",
    "resnet50": "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
    "resnet101": "https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth",
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, bias=False, stride=stride
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False, stride=1
        )
        self.bn2 = nn.BatchNorm2d(planes)
        # self.downsample = downsample
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))

        out = self.relu2(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))

        if self.downsample != None:
            residual = self.downsample(x)
        else:
            residual = x
        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            cfg,
            block_type,
            num_blocks,
            last_layer_stride=2,
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.block = block_type
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(
            num_blocks[1], 128, stride=2
        )
        self.layer3 = self._make_layer(
            num_blocks[2], 256, stride=2
        )
        self.layer4 = self._make_layer(
            num_blocks[3],
            512,
            stride=last_layer_stride,
        )

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "fc" not in k and "classifier" not in k:
                k = k.replace("backbone.", "")
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for now_stride in strides:
            layers.append(
                self.block(
                    self.inplanes, planes, stride=now_stride
                )
            )
            self.inplanes = planes * self.block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


class BBN_ResNet(nn.Module):
    def __init__(
            self,
            cfg,
            block_type,
            num_blocks,
            last_layer_stride=2,
    ):
        # cfg,
        # BottleNeck,
        # [3, 4, 6, 3],
        super(BBN_ResNet, self).__init__()
        self.inplanes = 64
        self.block = block_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], 64)
        self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
        # self.layer4 = self._make_layer(num_blocks[3] - 1, 512, stride=last_layer_stride)
        self.layer4 = self._make_layer(num_blocks[3], 512, stride=last_layer_stride)

        self.cb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)
        self.rb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)

    def load_model(self, pretrain):
        print("Loading Backbone pretrain model from {}......".format(pretrain))
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain)
        pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
        from collections import OrderedDict

        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                k = k[7:]
            if "fc" not in k and "classifier" not in k:
                k = k.replace("backbone.", "")
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        print("Backbone model has been loaded......")

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for now_stride in strides:
            layers.append(self.block(self.inplanes, planes, stride=now_stride))
            self.inplanes = planes * self.block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if "feature_cb" in kwargs:
            out = self.cb_block(out)
            return out
        elif "feature_rb" in kwargs:
            out = self.rb_block(out)
            return out
        out1 = self.cb_block(out)
        out2 = self.rb_block(out)
        out = torch.cat((out1, out2), dim=1)

        return out


def res50(
        cfg,
        pretrain=True,
        pretrained_model="/data/Data/pretrain_models/resnet50-19c8e357.pth",
        last_layer_stride=2,
):
    resnet = ResNet(
        cfg,
        BottleNeck,
        [3, 4, 6, 3],
        last_layer_stride=last_layer_stride,
    )
    if pretrain and pretrained_model != "":
        resnet.load_model(pretrain=pretrained_model)
    else:
        print("Choose to train from scratch")
    return resnet


class BBN_ResNextNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(BBN_ResNextNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # BBN
        self.cb_block = block(self.inplanes, self.inplanes // 4, stride=1)
        self.rb_block = block(self.inplanes, self.inplanes // 4, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion),
                                       )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.reshape(x.size(0), -1)
        # x = self.fc(x)
        if "feature_cb" in kwargs:
            out = self.cb_block(x)
            # print("feature_cb: ", out.shape)
            return out
        elif "feature_rb" in kwargs:
            out = self.rb_block(x)
            # print("feature_rb: ", out.shape)
            return out
        out1 = self.cb_block(x)
        out2 = self.rb_block(x)
        out = torch.cat((out1, out2), dim=1)
        # print('feature', out.shape)
        return out
    # def __init__(
    #         self,
    #         cfg,
    #         block_type,
    #         num_blocks,
    #         last_layer_stride=2,
    # ):
    #     # cfg,
    #     # BottleNeck,
    #     # [3, 4, 6, 3],
    #     super(BBN_ResNext101Net, self).__init__()
    #     self.inplanes = 64
    #     self.block = block_type
    #
    #     self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #     self.bn1 = nn.BatchNorm2d(64)
    #     self.relu = nn.ReLU(True)
    #     self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    #
    #     self.layer1 = self._make_layer(num_blocks[0], 64)
    #     self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
    #     self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
    #     # self.layer4 = self._make_layer(num_blocks[3] - 1, 512, stride=last_layer_stride)
    #     self.layer4 = self._make_layer(num_blocks[3], 512, stride=last_layer_stride)
    #
    #     self.cb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)
    #     self.rb_block = self.block(self.inplanes, self.inplanes // 4, stride=1)
    #
    # def load_model(self, pretrain):
    #     print("Loading Backbone pretrain model from {}......".format(pretrain))
    #     model_dict = self.state_dict()
    #     pretrain_dict = torch.load(pretrain)
    #     pretrain_dict = pretrain_dict["state_dict"] if "state_dict" in pretrain_dict else pretrain_dict
    #     from collections import OrderedDict
    #
    #     new_dict = OrderedDict()
    #     for k, v in pretrain_dict.items():
    #         if k.startswith("module"):
    #             k = k[7:]
    #         if "fc" not in k and "classifier" not in k:
    #             k = k.replace("backbone.", "")
    #             new_dict[k] = v
    #
    #     model_dict.update(new_dict)
    #     self.load_state_dict(model_dict)
    #     print("Backbone model has been loaded......")
    #
    # def _make_layer(self, num_block, planes, stride=1):
    #     strides = [stride] + [1] * (num_block - 1)
    #     layers = []
    #     for now_stride in strides:
    #         layers.append(self.block(self.inplanes, planes, stride=now_stride))
    #         self.inplanes = planes * self.block.expansion
    #     return nn.Sequential(*layers)
    #
    # def forward(self, x, **kwargs):
    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #     out = self.pool(out)
    #
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #
    #     if "feature_cb" in kwargs:
    #         out = self.cb_block(out)
    #         return out
    #     elif "feature_rb" in kwargs:
    #         out = self.rb_block(out)
    #         return out
    #     out1 = self.cb_block(out)
    #     out2 = self.rb_block(out)
    #     out = torch.cat((out1, out2), dim=1)
    #
    #     return out


def bbn_res50(
        cfg,
        pretrain=True,
        pretrained_model="/data/Data/pretrain_models/resnet50-19c8e357.pth",
        last_layer_stride=2,):
    resnet = BBN_ResNet(
        cfg,
        BottleNeck,
        [3, 4, 6, 3],
        last_layer_stride=last_layer_stride,
    )
    if pretrain and pretrained_model != "":
        resnet.load_model(pretrain=pretrained_model)
    else:
        print("Choose to train from scratch")
    return resnet


def _resnext(arch, cfg, test, block, layers, pretrained, progress, **kwargs):
    model = BBN_ResNextNet(block, layers, **kwargs)
    if not test:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(cfg.BACKBONE.PRETRAINED_MODEL)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

