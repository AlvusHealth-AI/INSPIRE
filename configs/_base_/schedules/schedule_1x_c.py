# optimizer
optimizer = dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-3,
    step=[16,19])

# optimizer = dict(type='AdamW', lr=1e-4)
# optimizer_config = dict(grad_clip=dict(max_norm=0.5, norm_type=2))
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
#     warmup_iters=500,
#     warmup_ratio=1e-3,
#     step=[6,11])

runner = dict(type='EpochBasedRunner', max_epochs=20)

