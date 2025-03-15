# optimizer
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# optimizer = dict(type='AdamW', lr=1e-4)
# optimizer_config = dict(grad_clip=dict(max_norm=3.689, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[15,18])

# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1e-3,
#     min_lr_ratio=1e-3,
#     by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=20)
