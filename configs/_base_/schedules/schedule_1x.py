# optimizer
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1e-3,
#     min_lr_ratio=0.015,
#     by_epoch=True)

# optimizer = dict(type='AdamW', lr=5e-5)
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1e-3,
#     min_lr_ratio=1e-2,
#     by_epoch=False)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-3,
    step=[16,19])
# learning policy
runner = dict(type='EpochBasedRunner', max_epochs=20)

