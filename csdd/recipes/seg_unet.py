from __future__ import annotations

import gin

from csdd.palette import default_palette


@gin.configurable
def build_seg_cfg(
    *,
    data_root: str = "data/CSDD_seg",
    work_dir: str = "work_dirs/seg",
    classes: list[str] = ("background", "Scratches", "Spots", "Rusts"),
    img_size: tuple[int, int] = (1024, 1024),
    max_epochs: int = 80,
    batch_size: int = 2,
    num_workers: int = 2,
    lr: float = 0.0005,
    seed: int = 42,
) -> dict:
    """MMSegmentation U-Net baseline config for CSDD."""
    num_classes = len(classes)
    metainfo = {
        "classes": tuple(classes),
        # Default palette; change if you need consistent colors for your report.
        "palette": default_palette(num_classes),
    }

    train_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations"},
        {"type": "Resize", "scale": img_size, "keep_ratio": False},
        {"type": "RandomFlip", "prob": 0.5},
        {"type": "PackSegInputs"},
    ]
    test_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "Resize", "scale": img_size, "keep_ratio": False},
        {"type": "LoadAnnotations"},
        {"type": "PackSegInputs"},
    ]

    dataset_common = dict(
        type="CSDDSegDataset",
        data_root=data_root,
        metainfo=metainfo,
    )

    cfg: dict = dict(
        default_scope="mmseg",
        work_dir=work_dir,
        randomness=dict(seed=seed),
        model=dict(
            type="EncoderDecoder",
            data_preprocessor=dict(
                type="SegDataPreProcessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                size=img_size,
                pad_val=0,
                seg_pad_val=255,
            ),
            backbone=dict(
                type="UNet",
                in_channels=3,
                base_channels=64,
                num_stages=5,
                strides=(1, 1, 1, 1, 1),
                enc_num_convs=(2, 2, 2, 2, 2),
                dec_num_convs=(2, 2, 2, 2),
                downsamples=(True, True, True, True),
                enc_dilations=(1, 1, 1, 1, 1),
                dec_dilations=(1, 1, 1, 1),
                norm_cfg=dict(type="BN", requires_grad=True),
                act_cfg=dict(type="ReLU"),
            ),
            decode_head=dict(
                type="FCNHead",
                in_channels=64,
                channels=64,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=dict(type="BN", requires_grad=True),
                align_corners=False,
                loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            ),
            auxiliary_head=None,
            train_cfg=dict(),
            test_cfg=dict(mode="whole"),
        ),
        train_dataloader=dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            sampler=dict(type="DefaultSampler", shuffle=True),
            dataset=dict(
                **dataset_common,
                split="train",
                pipeline=train_pipeline,
            ),
        ),
        val_dataloader=dict(
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=dict(
                **dataset_common,
                split="val",
                pipeline=test_pipeline,
                test_mode=True,
            ),
        ),
        test_dataloader=dict(
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=dict(
                **dataset_common,
                split="test",
                pipeline=test_pipeline,
                test_mode=True,
            ),
        ),
        val_evaluator=dict(type="IoUMetric", iou_metrics=["mIoU"], classwise=True),
        test_evaluator=dict(type="IoUMetric", iou_metrics=["mIoU"], classwise=True),
        optim_wrapper=dict(type="OptimWrapper", optimizer=dict(type="AdamW", lr=lr, weight_decay=0.01)),
        param_scheduler=[
            dict(type="PolyLR", eta_min=1e-6, power=0.9, begin=0, end=max_epochs, by_epoch=True),
        ],
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1),
        val_cfg=dict(type="ValLoop"),
        test_cfg=dict(type="TestLoop"),
        default_hooks=dict(
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", interval=50),
            param_scheduler=dict(type="ParamSchedulerHook"),
            checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="mIoU", rule="greater"),
            sampler_seed=dict(type="DistSamplerSeedHook"),
            visualization=dict(type="SegVisualizationHook"),
        ),
        log_processor=dict(type="LogProcessor", window_size=50, by_epoch=True),
        visualizer=dict(type="SegLocalVisualizer"),
    )

    return cfg
