data_preprocessor = dict(
    pad_val=0, seg_pad_val=0, size_divisor=32, type='SegDataPreProcessor')
data_root = 'data/OCTA_6m'
dataset_type = 'OCTA6mDataest'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(ch_in=1, ch_out=32, type='FRNet'),
    data_preprocessor=dict(
        pad_val=0, seg_pad_val=0, size_divisor=32, type='SegDataPreProcessor'),
    decode_head=dict(
        channels=32,
        concat_input=False,
        ignore_index=-1,
        in_channels=32,
        in_index=0,
        kernel_size=11,
        loss_decode=[
            dict(loss_name='loss_dice', loss_weight=1.0, type='DiceLoss'),
        ],
        num_classes=2,
        num_convs=1,
        out_channels=2,
        type='FCNHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(
    normalized_shape=[
        3,
    ], type='LN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        data_root='data/OCTA_6m',
        pipeline=[
            dict(
                color_type='unchanged',
                to_float32=True,
                type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='OCTA6mDataest'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mDice',
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(color_type='unchanged', to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            data_prefix=dict(
                img_path='images/train', seg_map_path='annotations/train'),
            data_root='data/OCTA_6m',
            pipeline=[
                dict(
                    color_type='unchanged',
                    to_float32=True,
                    type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='PackSegInputs'),
            ],
            type='OCTA6mDataest'),
        times=40000,
        type='RepeatDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(color_type='unchanged', to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(color_type='unchanged', to_float32=True, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val'),
        data_root='data/OCTA_6m',
        pipeline=[
            dict(
                color_type='unchanged',
                to_float32=True,
                type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='OCTA6mDataest'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mDice',
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='WandbVisBackend'),
    ])
work_dir = './work_dirs/frnet-config'
