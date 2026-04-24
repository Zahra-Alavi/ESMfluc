#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_loss_curve(run_folder, train_losses, val_losses):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    if len(val_losses) > 0:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig(f"{run_folder}/loss_curve.png")


def save_evaluation_outputs(run_folder, cls_report, conf_matrix, args):
    cls_report_df = pd.DataFrame(cls_report).transpose()
    latex_table = cls_report_df.to_latex(float_format="%.2f")
    print(cls_report)
    print(latex_table)
    print(conf_matrix)

    with open(f"{run_folder}/classification_report.txt", "w") as f:
        f.write(str(cls_report))
    with open(f"{run_folder}/classification_report.tex", "w") as f:
        f.write(latex_table)

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(f"{run_folder}/confusion_matrix.png")
    plt.close()

    with open(f"{run_folder}/args.txt", "w") as f:
        f.write(str(args))


def save_metrics_json(
    run_folder,
    args,
    on_cuda,
    total_seconds,
    epoch_times,
    gpu_epoch_peaks,
    gpu_overall_peak,
    cpu_rss_start,
    cpu_rss_end,
):
    metrics = {
        "device": str(args.device),
        "seed": args.seed,
        "amp_enabled": bool(args.mixed_precision and on_cuda),
        "total_seconds": total_seconds,
        "epoch_seconds": epoch_times,
        "gpu_peak_bytes_per_epoch": gpu_epoch_peaks if on_cuda else None,
        "gpu_overall_peak_bytes": gpu_overall_peak,
        "cpu_rss_start_bytes": cpu_rss_start,
        "cpu_rss_end_bytes": cpu_rss_end,
        "epochs_ran": len(epoch_times),
    }
    with open(f"{run_folder}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {run_folder}/metrics.json")


def save_run_summary_csv(
    run_folder,
    args,
    cls_report,
    conf_matrix,
    total_seconds,
    epoch_times,
    gpu_overall_peak,
    cpu_rss_start,
    cpu_rss_end,
    on_cuda,
):
    row = {
        "run_dir": run_folder,
        "batch_size": args.batch_size,
        "embedding_model": args.esm_model,
        "model_architecture": args.architecture,
        "loss_mode": args.loss_mode,
        "loss_function": args.loss_function,
        "head": args.head,
        "loss_desc": f"{args.loss_mode}|{args.loss_function}|head={args.head}",
        "total_time_seconds": total_seconds,
        "epochs_ran": len(epoch_times),
        "gpu_overall_peak_bytes": gpu_overall_peak,
        "cpu_rss_start_bytes": cpu_rss_start,
        "cpu_rss_end_bytes": cpu_rss_end,
        "cpu_rss_delta_bytes": (
            int(cpu_rss_end - cpu_rss_start)
            if (cpu_rss_end is not None and cpu_rss_start is not None)
            else None
        ),
        "seed": getattr(args, "seed", None),
        "device": str(args.device),
        "amp_enabled": bool(args.mixed_precision and on_cuda),
    }

    for key, val in cls_report.items():
        if key == "accuracy":
            row["accuracy"] = val
            continue
        if isinstance(val, dict):
            key_safe = str(key).replace(" ", "_")
            for subk, subval in val.items():
                subk_safe = subk.replace("-", "_")
                row[f"{key_safe}_{subk_safe}"] = subval

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            row[f"cm_true_{i}_pred_{j}"] = int(conf_matrix[i, j])

    summary_df = pd.DataFrame([row])
    summary_csv_path = f"{run_folder}/run_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved run summary to {summary_csv_path}")
