from __future__ import annotations

from pathlib import Path

import gin


@gin.configurable
def build_det_cfg(
    *,
    data_root: str = "data/CSDD_det",
    work_dir: str = "work_dirs/det",
    classes: list[str] = ("Scratches", "Spots", "Rusts"),
    img_size: tuple[int, int] = (1024, 1024),
    # RPN anchor scales are multiplied by feature stride. With FPN strides
    # [4, 8, 16, 32, 64] and default scale 8, the smallest anchor is 32px on P2,
    # which is often too large for tiny "Spots". Adding 2/4 enables 8/16px anchors on P2.
    rpn_scales: tuple[int, ...] = (2, 4, 8),
    # Class imbalance handling for the RCNN classification head (foreground classes only).
    # NOTE: For softmax CE in MMDet, bbox_head has an extra background logit, so we append
    # `rcnn_background_weight` automatically.
    rcnn_class_weights: tuple[float, ...] = (1.0, 3.0, 2.2),
    rcnn_background_weight: float = 1.0,
    max_epochs: int = 24,
    batch_size: int = 1,
    num_workers: int = 2,
    lr: float = 0.01,
    seed: int = 42,
) -> dict:
    """MMDetection Faster R-CNN (R50-FPN) baseline config for CSDD."""
    # Use absolute paths to avoid MMEngine joining `data_root` + `ann_file` twice on Windows.
    data_root_p = Path(data_root).resolve()
    ann_dir = (data_root_p / "annotations").resolve()

    metainfo = {"classes": tuple(classes)}
    if len(rcnn_class_weights) != len(classes):
        raise ValueError(f"rcnn_class_weights must have len(classes)={len(classes)}, got {len(rcnn_class_weights)}")
    rcnn_class_weight_with_bg = list(rcnn_class_weights) + [float(rcnn_background_weight)]

    train_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "LoadAnnotations", "with_bbox": True},
        {"type": "Resize", "scale": img_size, "keep_ratio": False},
        {"type": "RandomFlip", "prob": 0.5},
        {"type": "PackDetInputs"},
    ]
    test_pipeline = [
        {"type": "LoadImageFromFile"},
        {"type": "Resize", "scale": img_size, "keep_ratio": False},
        {"type": "LoadAnnotations", "with_bbox": True},
        {"type": "PackDetInputs", "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")},
    ]

    train_dataset = {
        "type": "CocoDataset",
        "data_root": str(data_root_p),
        "ann_file": str((ann_dir / "instances_train2017.json").resolve()),
        "data_prefix": {"img": "images/train2017/"},
        "metainfo": metainfo,
        "pipeline": train_pipeline,
    }
    val_dataset = {
        "type": "CocoDataset",
        "data_root": str(data_root_p),
        "ann_file": str((ann_dir / "instances_val2017.json").resolve()),
        "data_prefix": {"img": "images/val2017/"},
        "metainfo": metainfo,
        "pipeline": test_pipeline,
        "test_mode": True,
    }
    test_dataset = {
        "type": "CocoDataset",
        "data_root": str(data_root_p),
        "ann_file": str((ann_dir / "instances_test2017.json").resolve()),
        "data_prefix": {"img": "images/test2017/"},
        "metainfo": metainfo,
        "pipeline": test_pipeline,
        "test_mode": True,
    }

    cfg: dict = dict(
        default_scope="mmdet",
        work_dir=work_dir,
        randomness=dict(seed=seed),
        # Model (Faster R-CNN, standard baseline)
        model=dict(
            type="FasterRCNN",
            data_preprocessor=dict(
                type="DetDataPreprocessor",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
                pad_size_divisor=32,
            ),
            backbone=dict(
                type="ResNet",
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type="BN", requires_grad=True),
                norm_eval=True,
                style="pytorch",
                init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
            ),
            neck=dict(
                type="FPN",
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5,
            ),
            rpn_head=dict(
                type="RPNHead",
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type="AnchorGenerator",
                    scales=list(rpn_scales),
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64],
                ),
                bbox_coder=dict(type="DeltaXYWHBBoxCoder", target_means=[0.0] * 4, target_stds=[1.0] * 4),
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type="L1Loss", loss_weight=1.0),
            ),
            roi_head=dict(
                type="StandardRoIHead",
                bbox_roi_extractor=dict(
                    type="SingleRoIExtractor",
                    roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32],
                ),
                bbox_head=dict(
                    type="Shared2FCBBoxHead",
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=len(classes),
                    bbox_coder=dict(
                        type="DeltaXYWHBBoxCoder",
                        target_means=[0.0] * 4,
                        target_stds=[0.1, 0.1, 0.2, 0.2],
                    ),
                    reg_class_agnostic=False,
                    loss_cls=dict(
                        type="CrossEntropyLoss",
                        use_sigmoid=False,
                        class_weight=rcnn_class_weight_with_bg,
                        loss_weight=1.0,
                    ),
                    loss_bbox=dict(type="L1Loss", loss_weight=1.0),
                ),
            ),
            train_cfg=dict(
                rpn=dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.3,
                        match_low_quality=True,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(type="RandomSampler", num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False,
                ),
                rpn_proposal=dict(
                    nms_pre=2000,
                    max_per_img=1000,
                    nms=dict(type="nms", iou_threshold=0.7),
                    min_bbox_size=0,
                ),
                rcnn=dict(
                    assigner=dict(
                        type="MaxIoUAssigner",
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1,
                    ),
                    sampler=dict(type="RandomSampler", num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
                    pos_weight=-1,
                    debug=False,
                ),
            ),
            test_cfg=dict(
                rpn=dict(nms_pre=1000, max_per_img=1000, nms=dict(type="nms", iou_threshold=0.7), min_bbox_size=0),
                rcnn=dict(score_thr=0.05, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
            ),
        ),
        # Dataloaders
        train_dataloader=dict(
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            sampler=dict(type="DefaultSampler", shuffle=True),
            dataset=train_dataset,
        ),
        val_dataloader=dict(
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=False,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=val_dataset,
        ),
        test_dataloader=dict(
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            drop_last=False,
            sampler=dict(type="DefaultSampler", shuffle=False),
            dataset=test_dataset,
        ),
        # Evaluators (classwise -> per-class AP)
        val_evaluator=dict(
            type="CocoMetric",
            ann_file=str((ann_dir / "instances_val2017.json").resolve()),
            metric="bbox",
            classwise=True,
        ),
        test_evaluator=dict(
            type="CocoMetric",
            ann_file=str((ann_dir / "instances_test2017.json").resolve()),
            metric="bbox",
            classwise=True,
        ),
        # Optim / schedule
        optim_wrapper=dict(
            type="OptimWrapper",
            optimizer=dict(type="SGD", lr=lr, momentum=0.9, weight_decay=0.0001),
            clip_grad=None,
        ),
        param_scheduler=[
            dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
            dict(type="MultiStepLR", by_epoch=True, milestones=[int(max_epochs * 0.67), int(max_epochs * 0.89)], gamma=0.1),
        ],
        train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1),
        val_cfg=dict(type="ValLoop"),
        test_cfg=dict(type="TestLoop"),
        # Hooks
        default_hooks=dict(
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", interval=50),
            param_scheduler=dict(type="ParamSchedulerHook"),
            checkpoint=dict(
                type="CheckpointHook",
                interval=1,
                max_keep_ckpts=3,
                save_best="coco/bbox_mAP",
                rule="greater",
            ),
            sampler_seed=dict(type="DistSamplerSeedHook"),
            visualization=dict(type="DetVisualizationHook"),
        ),
        log_processor=dict(type="LogProcessor", window_size=50, by_epoch=True),
        env_cfg=dict(cudnn_benchmark=False),
        visualizer=dict(type="DetLocalVisualizer"),
    )

    return cfg
