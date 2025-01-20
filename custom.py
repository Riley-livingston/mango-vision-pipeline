"""
RTMDet Configuration File

This configuration file defines the settings and parameters for the RTMDet (Real-Time Detection) model,
a high-performance object detection framework. It includes settings for:
- Model architecture (backbone, neck, head)
- Training parameters (learning rate, batch size, epochs)
- Data preprocessing and augmentation
- Testing and validation configurations

The model is configured for card detection with a single class ('card').
"""

_backend_args = None
_multiscale_resize_transforms = [
    dict(
        transforms=[
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                320,
                320,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    320,
                    320,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
    dict(
        transforms=[
            dict(scale=(
                960,
                960,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    960,
                    960,
                ),
                type='LetterResize'),
        ],
        type='Compose'),
]
backend_args = None
base_lr = 0.004
batch_shapes_cfg = dict(
    batch_size=32,
    extra_pad_ratio=0.5,
    img_size=640,
    size_divisor=32,
    type='BatchShapePolicy')
class_name = ('card', )
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        strict_load=False,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=30,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                resize_type='mmdet.Resize',
                scale=(
                    640,
                    640,
                ),
                type='mmdet.RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='mmdet.RandomCrop'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='mmdet.Pad'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='mmdet.PipelineSwitchHook'),
]
data_root = '/content/drive/MyDrive/RTMDet_Images/'
dataset_type = 'YOLOv5CocoDataset'
deepen_factor = 1.0
default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmyolo'
dsl_topk = 13
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    640,
    640,
)
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
loss_bbox_weight = 2.0
loss_cls_weight = 1.0
lr_start_factor = 1e-05
max_epochs = 50
max_keep_ckpts = 3
metainfo = dict(
    classes=('card', ), palette=[
        (
            20,
            220,
            60,
        ),
    ])
mixup_max_cached_images = 20
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=1.0,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        type='CSPNeXt',
        widen_factor=1.0),
    bbox_head=dict(
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        head_module=dict(
            act_cfg=dict(inplace=True, type='SiLU'),
            feat_channels=256,
            featmap_strides=[
                8,
                16,
                32,
            ],
            in_channels=256,
            norm_cfg=dict(type='BN'),
            num_classes=1,
            pred_kernel_size=1,
            share_conv=True,
            stacked_convs=2,
            type='RTMDetSepBNHeadModule'),
        loss_bbox=dict(loss_weight=2.0, type='mmdet.GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='mmdet.QualityFocalLoss',
            use_sigmoid=True),
        prior_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='mmdet.MlvlPointGenerator'),
        type='RTMDetHead'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='YOLOv5DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        deepen_factor=1.0,
        expand_ratio=0.5,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(type='BN'),
        num_csp_blocks=3,
        out_channels=256,
        type='CSPNeXtPAFPN',
        widen_factor=1.0),
    test_cfg=dict(
        max_per_img=30,
        multi_label=True,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=15000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            iou_calculator=dict(type='mmdet.BboxOverlaps2D'),
            num_classes=1,
            topk=13,
            type='BatchDynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='YOLODetector')
model_test_cfg = dict(
    max_per_img=30,
    multi_label=True,
    nms=dict(iou_threshold=0.7, type='nms'),
    nms_pre=15000,
    score_thr=0.001)
mosaic_max_cached_images = 40
norm_cfg = dict(type='BN')
num_classes = 1
num_epochs_stage2 = 20
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=25,
        begin=25,
        by_epoch=True,
        convert_to_iter_based=True,
        end=50,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
persistent_workers = True
qfl_beta = 2.0
random_resize_ratio_range = (
    0.1,
    2.0,
)
resume = False
save_checkpoint_intervals = 10
strides = [
    8,
    16,
    32,
]
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='val_labels.json',
        batch_shapes_cfg=dict(
            batch_size=32,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
        data_prefix=dict(img='val_images/'),
        data_root='/content/drive/MyDrive/RTMDet_Images/',
        metainfo=dict(classes=('card', ), palette=[
            (
                20,
                220,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/content/drive/MyDrive/RTMDet_Images/val_labels.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(scale=(
        640,
        640,
    ), type='YOLOv5KeepRatioResize'),
    dict(
        allow_scale_up=False,
        pad_val=dict(img=114),
        scale=(
            640,
            640,
        ),
        type='LetterResize'),
    dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'pad_param',
        ),
        type='mmdet.PackDetInputs'),
]
train_ann_file = 'train_labels.json'
train_batch_size_per_gpu = 4
train_cfg = dict(
    dynamic_intervals=[
        (
            30,
            1,
        ),
    ],
    max_epochs=50,
    type='EpochBasedTrainLoop',
    val_interval=10)
train_data_prefix = 'train_images/'
train_dataloader = dict(
    batch_size=4,
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict(
        ann_file='train_labels.json',
        data_prefix=dict(img='train_images/'),
        data_root='/content/drive/MyDrive/RTMDet_Images/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=('card', ), palette=[
            (
                20,
                220,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=40,
                pad_val=114.0,
                type='Mosaic',
                use_cached=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                resize_type='mmdet.Resize',
                scale=(
                    1280,
                    1280,
                ),
                type='mmdet.RandomResize'),
            dict(crop_size=(
                640,
                640,
            ), type='mmdet.RandomCrop'),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(prob=0.5, type='mmdet.RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='mmdet.Pad'),
            dict(max_cached_images=20, type='YOLOv5MixUp', use_cached=True),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='YOLOv5CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_num_workers = 4
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        img_scale=(
            640,
            640,
        ),
        max_cached_images=40,
        pad_val=114.0,
        type='Mosaic',
        use_cached=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        resize_type='mmdet.Resize',
        scale=(
            1280,
            1280,
        ),
        type='mmdet.RandomResize'),
    dict(crop_size=(
        640,
        640,
    ), type='mmdet.RandomCrop'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            640,
            640,
        ),
        type='mmdet.Pad'),
    dict(max_cached_images=20, type='YOLOv5MixUp', use_cached=True),
    dict(type='mmdet.PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        resize_type='mmdet.Resize',
        scale=(
            640,
            640,
        ),
        type='mmdet.RandomResize'),
    dict(crop_size=(
        640,
        640,
    ), type='mmdet.RandomCrop'),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(prob=0.5, type='mmdet.RandomFlip'),
    dict(
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            640,
            640,
        ),
        type='mmdet.Pad'),
    dict(type='mmdet.PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.65, type='nms')),
    type='mmdet.DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    transforms=[
                        dict(scale=(
                            640,
                            640,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                640,
                                640,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            320,
                            320,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                320,
                                320,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
                dict(
                    transforms=[
                        dict(scale=(
                            960,
                            960,
                        ), type='YOLOv5KeepRatioResize'),
                        dict(
                            allow_scale_up=False,
                            pad_val=dict(img=114),
                            scale=(
                                960,
                                960,
                            ),
                            type='LetterResize'),
                    ],
                    type='Compose'),
            ],
            [
                dict(prob=1.0, type='mmdet.RandomFlip'),
                dict(prob=0.0, type='mmdet.RandomFlip'),
            ],
            [
                dict(type='mmdet.LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'pad_param',
                        'flip',
                        'flip_direction',
                    ),
                    type='mmdet.PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_ann_file = 'val_labels.json'
val_batch_size_per_gpu = 32
val_cfg = dict(type='ValLoop')
val_data_prefix = 'val_images/'
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='val_labels.json',
        batch_shapes_cfg=dict(
            batch_size=32,
            extra_pad_ratio=0.5,
            img_size=640,
            size_divisor=32,
            type='BatchShapePolicy'),
        data_prefix=dict(img='val_images/'),
        data_root='/content/drive/MyDrive/RTMDet_Images/',
        metainfo=dict(classes=('card', ), palette=[
            (
                20,
                220,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(scale=(
                640,
                640,
            ), type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                ),
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='YOLOv5CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/content/drive/MyDrive/RTMDet_Images/val_labels.json',
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='mmdet.CocoMetric')
val_interval_stage2 = 1
val_num_workers = 10
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='mmdet.DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
weight_decay = 0.05
widen_factor = 1.0
work_dir = '/content/drive/MyDrive/RTMDet_Models/logs_v11'
