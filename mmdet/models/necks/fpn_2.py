from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule
from mmcv.cnn import constant_init, xavier_init
from mmcv.runner import BaseModule, auto_fp16
from mmcv.ops.deform_conv import DeformConv2dPack as DeformConv2d 
# from mmcv.ops.deform_conv import DeformConv2d 
from ..builder import NECKS




class ISE(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(ISE, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False,).to('cuda')
        self.batchnorm_in = nn.BatchNorm2d(in_chan).to('cuda')
        self.batchnorm_out = nn.BatchNorm2d(out_chan).to('cuda')
        self.sigmoid = nn.Sigmoid()
        self.swish = nn.SiLU(inplace=False).to('cuda')
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False,).to('cuda')
        self.lateral = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False,).to('cuda')
        
        xavier_init(self.conv_atten,distribution='uniform')
        xavier_init(self.conv,distribution='uniform')
        xavier_init(self.lateral,distribution='uniform')

    def forward(self, x):
        temp = self.swish(self.batchnorm_in(self.conv_atten(F.avg_pool2d(x, x.size()[2:])).to('cuda')))
        atten = self.sigmoid(temp)
        feat = torch.mul(x, atten)
        x = self.swish(self.batchnorm_in(self.lateral(x))) + feat
        feat = self.swish(self.batchnorm_out(self.conv(x)))
        return feat


class ISA(nn.Module): 
    def __init__(self, in_nc=576, out_nc=288, norm=None):
        super(ISA, self).__init__()
        self.lateral_conv = ISE(in_nc, out_nc).to('cuda')
        self.up = nn.ConvTranspose2d(in_nc,in_nc, kernel_size=2, stride=2, padding=0, bias=False,)
        self.offset = nn.Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False,).to('cuda')
        self.lateral = nn.Conv2d(in_nc, out_nc, kernel_size=1, stride=1, padding=0, bias=False,).to('cuda')
        self.deform= DeformConv2d(2*out_nc, out_nc, kernel_size=3, stride=1, padding=1, dilation=1, deform_groups=1,bias=False).to('cuda')
        # self.deform= DeformConv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1, dilation=1,deform_groups=16,bias=False).to('cuda')
        self.swish = nn.SiLU(inplace=True)
        self.batchnorm=nn.BatchNorm2d(out_nc).to('cuda')
        self.batchnorm_up=nn.BatchNorm2d(in_nc).to('cuda')
        self.gate_weight=nn.Conv2d(2*out_nc,1,kernel_size=1, stride=1, padding=0, bias=True,).to('cuda')
        # self.gate_weight_1= nn.Conv2d(out_nc,1,kernel_size=1, stride=1, padding=0, bias=True,).to('cuda')
        # self.gate_weight_2= nn.Conv2d(out_nc,1,kernel_size=1, stride=1, padding=0, bias=True,).to('cuda')

        xavier_init(self.up,distribution='uniform')
        xavier_init(self.offset,distribution='uniform')
        xavier_init(self.deform,distribution='uniform')
        # xavier_init(self.gate_weight_1,distribution='uniform')
        # xavier_init(self.gate_weight_2,distribution='uniform')
        xavier_init(self.gate_weight,distribution='uniform')



    def forward(self, feat_l, feat_s):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            # feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False).to('cuda')
            feat_up = self.swish(self.batchnorm_up(self.up(feat_s.to('cuda'))))
        else:
            feat_up = feat_s.to('cuda')
        feat_up = self.swish(self.lateral(feat_up))
        feat_arm = self.lateral_conv(feat_l.to('cuda'))  # 0~1 * feats
        offset = self.batchnorm(self.offset(torch.cat([feat_arm, feat_up * 2], dim=1).to('cuda')))  # concat for offset by compute the dif

        # feat_align = self.swish(self.deform(feat_up, offset))  # [feat, offset]
        feat_align = self.swish(self.deform(torch.cat([feat_up, offset],dim=1)))  # [feat, offset]
        # add_weight = torch.sigmoid(self.gate_weight_1(feat_arm)+self.gate_weight_2(feat_align))
        add_weight = torch.tanh(self.gate_weight(torch.cat([feat_align, feat_arm],dim=1)))
        # return feat_align + feat_arm with weights
        return add_weight * feat_align + (1 - add_weight) * feat_arm

@NECKS.register_module()
class FPN_plusplus(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN_plusplus, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.l_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        self.c=180
        self.fpn_1=ISA(self.c,self.c).to('cuda')
        self.fpn_2=ISA(self.c,self.c).to('cuda')
        self.fpn_3=ISA(self.c,self.c).to('cuda')

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                self.c,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.c,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.fpn_convs.append(fpn_conv)
            self.lateral_convs.append(l_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        laterals[2] = self.fpn_1(laterals[2],laterals[3])
        laterals[1] = self.fpn_2(laterals[1],laterals[2])
        laterals[0] = self.fpn_3(laterals[0],laterals[1])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
