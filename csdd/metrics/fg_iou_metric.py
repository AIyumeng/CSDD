from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
from mmengine.dist import is_main_process
from mmengine.logging import MMLogger, print_log
from prettytable import PrettyTable

from mmseg.evaluation.metrics import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class CSDDFgIoUMetric(IoUMetric):
    """IoU metric that reports mean scores over foreground classes only.

    CSDD uses label 0 as background. For reporting, it's common to compute
    the mean IoU over defect classes (1..K) rather than including background.

    This metric keeps per-class tables (including background), but aggregates
    `mIoU/mAcc/...` over classes 1..K to match that convention.
    """

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            if self.output_dir:
                logger.info(f"results are saved to {self.output_dir}")
            return OrderedDict()

        results = tuple(zip(*results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])

        ret_metrics = self.total_area_to_metrics(
            total_area_intersect,
            total_area_union,
            total_area_pred_label,
            total_area_label,
            self.metrics,
            self.nan_to_num,
            self.beta,
        )

        class_names = self.dataset_meta["classes"]
        fg_slice = slice(1, None) if len(class_names) > 1 else slice(None)

        # Summary metrics
        metrics: Dict[str, float] = {}
        aAcc = float(np.round(ret_metrics["aAcc"] * 100, 2))
        metrics["aAcc"] = aAcc

        for key, val in ret_metrics.items():
            if key == "aAcc":
                continue
            # val is per-class array; aggregate over foreground classes only.
            mean_val = float(np.round(np.nanmean(val[fg_slice]) * 100, 2))
            metrics["m" + key] = mean_val

        # Per-class table (same as IoUMetric)
        per_class = OrderedDict(
            {
                k: np.round(v * 100, 2)
                for k, v in ret_metrics.items()
                if k != "aAcc"
            }
        )
        per_class.update({"Class": class_names})
        per_class.move_to_end("Class", last=False)

        class_table = PrettyTable()
        for k, v in per_class.items():
            class_table.add_column(k, v)

        print_log("per class results:", logger)
        print_log("\n" + class_table.get_string(), logger=logger)

        # If saving outputs, only main process writes.
        if self.output_dir and is_main_process():
            pass

        return metrics

