_base_ = './htc/htc_r50_fpn_1x_coco.py'


model = dict(
    backbone=dict(
        type='ResNet_back',
        conv_cfg=dict(type='ConvAWS'),
        output_img=True),
    neck=dict(
        type='FPN',
        steps=2,
        out_channels=64,
        dilations=(1, 3, 6, 1),
        backbone=dict(
            inplanes=256,
            type='ResNet_back',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            pretrained='torchvision://resnet50',
            style='pytorch')))
