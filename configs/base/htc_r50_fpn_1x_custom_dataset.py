_base_ = './htc_r50_fpn_1x_custom_dataset.py'
model = dict(
    roi_head=dict(
        semantic_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[8]),
        semantic_head=dict(
            type='FusedSemanticHead',
            num_ins=5,
            fusion_level=1,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=183,
            loss_seg=dict(
                type='CrossEntropyLoss', ignore_index=255, loss_weight=0.2))))
data_root = './data/custom_dataset/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=[(224, 224)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(224,224)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='Resize',img_scale=[(1333, 800),(1500,1000),(1800,1200),(2100,1400),(2400,1600)], multiscale_mode='value',keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(
        seg_prefix=data_root + 'train_data/',
        pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
