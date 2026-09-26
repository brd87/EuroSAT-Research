import torch
from torch import nn

from collections import Counter

import config
import pipeline_utilities.pipeline as pipeline
import pipeline_utilities.dataset_get as dataset_get
import pipeline_utilities.metrics as metrics
import reporting.report as report

from models.ConvNeXt import ConvNeXt
from models.EfficientNetV2 import EfficientNetV2
from models.ResNeXt import ResNeXt
# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def main():
    print("START")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------- DATA -----------------
    material =[
        dataset_get.eurosat_ms(),
        dataset_get.eurosat_rgb(),
        ]

    # ----------------- MODEL -----------------
    models = [
        ConvNeXt,
        EfficientNetV2,
        ResNeXt
        ]

    criterion = nn.CrossEntropyLoss()

    for dataset, ds_name in material:
        metrics_set = []
        in_ch = dataset[0][0].shape[0]      
        labels = [dataset[i][1] for i in range(len(dataset))]
        print(Counter(int(l) for l in labels))
        print("classes:", getattr(dataset, "classes", None))
        print("config.CLASSES:", config.CLASSES)
        for model_cls in models:
            model = model_cls(in_channels=in_ch)
            metrics_result = pipeline.ran(model, dataset, device, criterion, ds_name)
            metrics_set.append(metrics_result)

        metrics_merged = metrics.merge(metrics_set)
        metrics_merged_ranked = metrics.merged_rank(metrics_merged)

        report.build_excel_report(ds_name, metrics_merged)
        report.build_pdf_report(ds_name, metrics_merged, metrics_merged_ranked)
        report.build_png_dashboard(ds_name, metrics_merged, metrics_merged_ranked)


if __name__ == "__main__":
    main()

