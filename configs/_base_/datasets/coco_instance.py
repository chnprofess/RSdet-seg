# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True), #1333, 800
    dict(type='RandomFlip', prob=[0.5, 0.5], direction=['vertical', 'horizontal']),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(512,512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator

# train_pipeline = [
#     dict(type='Mosaic', img_scale=(512, 512), pad_val=114.0),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.1, 2),
#         # img_scale is (width, height)
#         border=(-img_scale[0] // 2, -img_scale[1] // 2)),
#     dict(
#         type='MixUp',
#         img_scale=(512, 512),
#         ratio_range=(0.8, 1.6),
#         pad_val=114.0),
#     dict(type='YOLOXHSVRandomAug'),
#     dict(type='RandomFlip', prob=0.5),
#     # According to the official implementation, multi-scale
#     # training is not considered here but in the
#     # 'mmdet/models/detectors/yolox.py'.
#     # Resize and Pad are for the last 15 epochs when Mosaic,
#     # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
#     dict(type='Resize', scale=(512, 512), keep_ratio=True),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
#     dict(type='PackDetInputs')
# ]

