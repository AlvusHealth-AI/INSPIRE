# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-3,
    min_lr_ratio=0.001,
    by_epoch=True)
# optimizer = dict(type='AdamW', lr=1e-4)
# optimizer_config = dict(grad_clip=dict(max_norm=3.689, norm_type=2))
# # learning policy
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1e-3,
#     min_lr_ratio=2e-2,
#     by_epoch=True)
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1e-3,
#     step=[7,10])

runner = dict(type='EpochBasedRunner', max_epochs=12)

