import torch
from torch import nn

import config
import pipeline_utilities.pipeline as pipeline
import pipeline_utilities.dataset_get as dataset_get
import pipeline_utilities.metrics as metrics
import reporting.report as report

from models.ConvNeXt import ConvNeXt
from models.EfficientNetV2 import EfficientNetV2
from models.ResNeXt import ResNeXt


def main():
    print("START")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------- DATA -----------------
    material =[
        dataset_get.eurosat_rgb(),
        dataset_get.eurosat_ms()
        ]

    # ----------------- MODEL -----------------
    models = [
        ConvNeXt,
        EfficientNetV2,
        ResNeXt
        ]

    criterion = nn.CrossEntropyLoss()

    for dataset, ds_type in material:
        metrics_set = []
        in_ch = dataset[0][0].shape[0]      

        for model_cls in models:
            model = model_cls(in_channels=in_ch)
            metrics_result = pipeline.ran(model, dataset, device, criterion)
            metrics_set.append(metrics_result)

        metrics_merged = metrics.merge(metrics_set)
        metrics_merged_ranked = metrics.merged_rank(metrics_merged)

        report.build_excel_report(metrics_merged, ds_type)
        report.build_pdf_report(metrics_merged, metrics_merged_ranked, ds_type)
        report.build_png_dashboard(metrics_merged, metrics_merged_ranked, ds_type)


if __name__ == "__main__":
    main()

