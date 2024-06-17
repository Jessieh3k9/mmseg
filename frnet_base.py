_base_ = [
    './configs/_base_/datasets/drive.py',
    './configs/_base_/default_runtime.py', './configs/_base_/schedules/schedule_40k.py'
]
norm_cfg = dict(type='LN', normalized_shape=[3])

crop_size = (64, 64)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        64,
        64,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='FRNet',
        # drop_path_rate=0.4,
        # layer_scale_init_value=1.0,
        ch_in=3,
        ch_out=3
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=1,
        kernel_size=11,
        num_convs=1,
        in_index=0,
        channels=4,
        num_classes=2,
        concat_input=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ]),

    # model training and testing settings
    test_cfg=dict(mode='whole'),
    train_cfg=dict()
)
# optimizer

param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=40000,
        eta_min=0.0001,
        power=0.9,
        type='PolyLR'),
]
vis_backends = [dict(type='LocalVisBackend')] #不使用wandb记录
# vis_backends = [dict(type='LocalVisBackend'),dict(type='WandbVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
