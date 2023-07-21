_base_ = [
    '../_base_/models/fdlnet.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_125k.py'
]
model = dict(
    backbone=dict(
        backbone_cfg=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnet18_v1c')),
                ),
)

lr_config = dict(warmup='linear', warmup_iters=1000)
optimizer = dict(lr=0.01)
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
)
