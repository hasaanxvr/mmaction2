auto_scale_lr = dict(base_batch_size=192, enable=False)
backend_args = None
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(interval=50, priority='VERY_LOW', type='CheckInvalidLossHook'),
]
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_size_divisor=1,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreprocessor')
data_root = 'data/coco/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
input_size = 320
launcher = 'none'
load_from = 'configs/ssd/checkpoints/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 3
model = dict(
    backbone=dict(
        init_cfg=dict(layer='Conv2d', std=0.03, type='TruncNormal'),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_indices=(
            4,
            7,
        ),
        type='MobileNetV2'),
    bbox_head=dict(
        act_cfg=dict(type='ReLU6'),
        anchor_generator=dict(
            max_sizes=[
                100,
                150,
                202,
                253,
                304,
                320,
            ],
            min_sizes=[
                48,
                100,
                150,
                202,
                253,
                304,
            ],
            ratios=[
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
            ],
            scale_major=False,
            strides=[
                16,
                32,
                64,
                107,
                160,
                320,
            ],
            type='SSDAnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        in_channels=(
            96,
            1280,
            512,
            256,
            256,
            128,
        ),
        init_cfg=dict(layer='Conv2d', std=0.001, type='Normal'),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        num_classes=80,
        type='SSDHead',
        use_depthwise=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(type='ReLU6'),
        in_channels=(
            96,
            1280,
        ),
        init_cfg=dict(layer='Conv2d', std=0.03, type='TruncNormal'),
        l2_norm_scale=None,
        level_paddings=(
            1,
            1,
            1,
            1,
        ),
        level_strides=(
            2,
            2,
            2,
            2,
        ),
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        out_channels=(
            96,
            1280,
            512,
            256,
            256,
            128,
        ),
        type='SSDNeck',
        use_depthwise=True),
    test_cfg=dict(
        max_per_img=200,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.45, type='nms'),
        nms_pre=1000,
        score_thr=0.02),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            gt_max_assign_all=False,
            ignore_iof_thr=-1,
            min_pos_iou=0.0,
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        neg_pos_ratio=3,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=1.0),
    type='SingleStageDetector')
optim_wrapper = dict(
    optimizer=dict(lr=0.015, momentum=0.9, type='SGD', weight_decay=4e-05),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        T_max=3,
        begin=0,
        by_epoch=True,
        end=3,
        eta_min=0,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                320,
                320,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        320,
        320,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=3, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=24,
    dataset=dict(
        dataset=dict(
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/'),
            data_root='data/coco/',
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    mean=[
                        123.675,
                        116.28,
                        103.53,
                    ],
                    ratio_range=(
                        1,
                        4,
                    ),
                    to_rgb=True,
                    type='Expand'),
                dict(
                    min_crop_size=0.3,
                    min_ious=(
                        0.1,
                        0.3,
                        0.5,
                        0.7,
                        0.9,
                    ),
                    type='MinIoURandomCrop'),
                dict(keep_ratio=False, scale=(
                    320,
                    320,
                ), type='Resize'),
                dict(prob=0.5, type='RandomFlip'),
                dict(
                    brightness_delta=32,
                    contrast_range=(
                        0.5,
                        1.5,
                    ),
                    hue_delta=18,
                    saturation_range=(
                        0.5,
                        1.5,
                    ),
                    type='PhotoMetricDistortion'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        times=5,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        ratio_range=(
            1,
            4,
        ),
        to_rgb=True,
        type='Expand'),
    dict(
        min_crop_size=0.3,
        min_ious=(
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
        ),
        type='MinIoURandomCrop'),
    dict(keep_ratio=False, scale=(
        320,
        320,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.5,
            1.5,
        ),
        hue_delta=18,
        saturation_range=(
            0.5,
            1.5,
        ),
        type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='annotations/instances_val2017.json',
        backend_args=None,
        data_prefix=dict(img='val2017/'),
        data_root='data/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                320,
                320,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/coco/annotations/instances_val2017.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'configs/ssd/training_dump'
