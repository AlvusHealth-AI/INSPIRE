# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import Sequential, load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from .resnet import BasicBlock
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class Bottleneck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 back_inplanes=None,
                 init_cfg=None,
                 **kwargs):
        super(Bottleneck, self).__init__(
            inplanes, planes, init_cfg=init_cfg, **kwargs)

        self.conv2 = build_conv_layer(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            bias=False)

        self.inplanes = inplanes
        if self.back_inplanes:
            self.back_conv = build_conv_layer(
                None,
                self.back_inplanes,
                planes * self.expansion,
                1,
                stride=1,
                bias=True)

    def back_forward(self, x, back_feat):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        if self.inplanes:
            back_feat = self.back_conv(back_feat)
            out = out + back_feat

        out = self.relu(out)

        return out


class ResLayer(Sequential):
    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 downsample_first=True,
                 back_inplanes=None,
                 **kwargs):
        self.block = block
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down and stride != 1:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                build_conv_layer(
                    conv_cfg,
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                build_norm_layer(norm_cfg, planes * block.expansion)[1]
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                back_inplanes=back_inplanes,
                **kwargs))
        inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    **kwargs))

        super(ResLayer, self).__init__(*layers)


@BACKBONES.register_module()
class ResNet_back(ResNet):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 stage=(False, True, True, True),
                 back_inplanes=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        self.pretrained = pretrained
        if init_cfg is not None:
            assert isinstance(init_cfg, dict), \
                f'init_cfg must be a dict, but got {type(init_cfg)}'
            if 'type' in init_cfg:
                assert init_cfg.get('type') == 'Pretrained', \
                    'Only can initialize module by loading a pretrained model'
            else:
                raise KeyError('`init_cfg` must contain the key "type"')
            self.pretrained = init_cfg.get('checkpoint')
        self.stage = stage
        self.back_inplanes = back_inplanes
        super(ResNet_back, self).__init__(**kwargs)

        self.inplanes = self.stem_channels
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            stage_plugins = self.make_stage_plugins(self.plugins, i)
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                back_inplanes=back_inplanes if i > 0 else None,
                plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    def forward(self, x):
        outs = list(super(ResNet_back, self).forward(x))
        if self.output_img:
            outs.insert(0, x)
        return tuple(outs)

    def back_forward(self, x, back_feats):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            back_feat = back_feats[i] if i > 0 else None
            for layer in res_layer:
                x = layer.back_forward(x, back_feat)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)