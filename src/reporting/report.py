from pathlib import Path

import reporting.excel_exporter as excel_exporter
import reporting.report_builder as report_builder
import config


def build_excel_report(ds_type, metrics_merged, path:Path=None, checkpoints=None):
    if not path: path = _path_maker("report.xlsx", ds_type)
    return excel_exporter.export(metrics_merged, path=path, checkpoints=checkpoints)


def build_pdf_report(ds_type, metrics_merged, metrics_merged_ranked, path:Path=None):
    if not path: path = _path_maker("summary_report.pdf", ds_type)
    return report_builder.build_pdf(metrics_merged, metrics_merged_ranked, path)


def build_png_dashboard(ds_type, metrics_merged, metrics_merged_ranked, path:Path=None, dpi=200):
    if not path: path = _path_maker("dashboard.png", ds_type)
    return report_builder.build_png(metrics_merged, metrics_merged_ranked, path, dpi=dpi)

def _path_maker(file_name, ds_type) -> Path:
    path = Path(__file__).resolve().parents[2] / f"experiments/{config.EXPERIMENT_NAME}/report{ds_type}"
    path.mkdir(parents = True, exist_ok = True)
    return path / file_name