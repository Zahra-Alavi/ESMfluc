#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 13 2026

Run classification experiments at different temperature thresholds on mdcath dataset.
This script automates running multiple experiments with different neq_thresholds.

Usage:
    python run_temperature_experiments.py --train_data train.csv --test_data test.csv --config experiments_config.json
"""

import argparse
import json
import subprocess
import os
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run classification experiments at different temperature thresholds."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data CSV file.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data CSV file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file with experiment parameters (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./temperature_experiments",
        help="Base directory for experiment outputs (default: ./temperature_experiments).",
    )
    parser.add_argument(
        "--esm_model",
        type=str,
        default="esm2_t12_35M_UR50D",
        help="ESM model to use (default: esm2_t12_35M_UR50D).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs (default: 20).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def get_default_temperature_configs():
    """
    Define default temperature threshold configurations to test.
    Each config represents different "temperatures" (neq_thresholds).
    """
    configs = [
        {
            "name": "binary_low_temp",
            "description": "Binary classification with threshold at 1.0 (rigid vs flexible)",
            "num_classes": 2,
            "neq_thresholds": [1.0],
        },
        {
            "name": "binary_mid_temp",
            "description": "Binary classification with higher threshold at 1.5",
            "num_classes": 2,
            "neq_thresholds": [1.5],
        },
        {
            "name": "binary_high_temp",
            "description": "Binary classification with threshold at 2.0",
            "num_classes": 2,
            "neq_thresholds": [2.0],
        },
        {
            "name": "three_class_low",
            "description": "3-class classification (rigid, moderate, flexible)",
            "num_classes": 3,
            "neq_thresholds": [1.0, 1.5],
        },
        {
            "name": "three_class_mid",
            "description": "3-class classification with different thresholds",
            "num_classes": 3,
            "neq_thresholds": [1.0, 2.0],
        },
        {
            "name": "four_class_standard",
            "description": "4-class classification (standard setup)",
            "num_classes": 4,
            "neq_thresholds": [1.0, 2.0, 4.0],
        },
        {
            "name": "four_class_fine",
            "description": "4-class classification with finer gradation",
            "num_classes": 4,
            "neq_thresholds": [1.0, 1.5, 2.5],
        },
        {
            "name": "five_class",
            "description": "5-class classification for fine-grained flexibility",
            "num_classes": 5,
            "neq_thresholds": [1.0, 1.5, 2.0, 3.0],
        },
    ]
    return configs


def load_config_from_file(config_path):
    """Load experiment configurations from JSON file."""
    with open(config_path, "r") as f:
        configs = json.load(f)
    
    # Validate configs
    for config in configs:
        if "name" not in config:
            raise ValueError("Each config must have a 'name' field")
        if "num_classes" not in config:
            raise ValueError(f"Config '{config['name']}' missing 'num_classes'")
        if "neq_thresholds" not in config:
            raise ValueError(f"Config '{config['name']}' missing 'neq_thresholds'")
        
        # Validate threshold count
        expected_thresholds = config["num_classes"] - 1
        if len(config["neq_thresholds"]) != expected_thresholds:
            raise ValueError(
                f"Config '{config['name']}': num_classes={config['num_classes']} "
                f"requires {expected_thresholds} thresholds, got {len(config['neq_thresholds'])}"
            )
    
    return configs


def run_experiment(config, train_data, test_data, output_dir, args):
    """
    Run a single experiment with given configuration.
    
    Args:
        config: Dictionary with experiment configuration
        train_data: Path to training data
        test_data: Path to test data
        output_dir: Base output directory
        args: Command-line arguments
    
    Returns:
        True if experiment succeeded, False otherwise
    """
    exp_name = config["name"]
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {exp_name}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Classes: {config['num_classes']}")
    print(f"Thresholds: {config['neq_thresholds']}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        "python", "main.py",
        "--train_data_file", train_data,
        "--test_data_file", test_data,
        "--num_classes", str(config["num_classes"]),
        "--neq_thresholds", *[str(t) for t in config["neq_thresholds"]],
        "--esm_model", args.esm_model,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--seed", str(args.seed),
        "--result_foldername", exp_dir,
    ]
    
    # Add any additional parameters from config
    if "architecture" in config:
        cmd.extend(["--architecture", config["architecture"]])
    if "hidden_size" in config:
        cmd.extend(["--hidden_size", str(config["hidden_size"])])
    if "num_layers" in config:
        cmd.extend(["--num_layers", str(config["num_layers"])])
    if "dropout" in config:
        cmd.extend(["--dropout", str(config["dropout"])])
    if "loss_function" in config:
        cmd.extend(["--loss_function", config["loss_function"]])
    if "lr" in config:
        cmd.extend(["--lr", str(config["lr"])])
    if "oversampling" in config and config["oversampling"]:
        cmd.append("--oversampling")
    if "mixed_precision" in config and config["mixed_precision"]:
        cmd.append("--mixed_precision")
    
    # Save experiment config
    config_file = os.path.join(exp_dir, "experiment_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved experiment config to {config_file}")
    
    # Run experiment
    print(f"[INFO] Running command: {' '.join(cmd)}")
    
    log_file = os.path.join(exp_dir, "training_log.txt")
    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    
    if result.returncode == 0:
        print(f"[OK] Experiment '{exp_name}' completed successfully")
        print(f"[INFO] Logs saved to {log_file}")
        return True
    else:
        print(f"[ERROR] Experiment '{exp_name}' failed with return code {result.returncode}")
        print(f"[INFO] Check logs at {log_file}")
        return False


def create_summary_report(output_dir, configs, results):
    """Create a summary report of all experiments."""
    report_path = os.path.join(output_dir, "experiment_summary.json")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(configs),
        "successful": sum(results),
        "failed": len(results) - sum(results),
        "experiments": [],
    }
    
    for config, success in zip(configs, results):
        summary["experiments"].append({
            "name": config["name"],
            "description": config.get("description", "N/A"),
            "num_classes": config["num_classes"],
            "neq_thresholds": config["neq_thresholds"],
            "status": "success" if success else "failed",
        })
    
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[OK] Summary report saved to {report_path}")
    return summary


def main():
    args = parse_args()
    
    # Load or use default configurations
    if args.config:
        print(f"[INFO] Loading experiment configs from {args.config}")
        configs = load_config_from_file(args.config)
    else:
        print("[INFO] Using default temperature configurations")
        configs = get_default_temperature_configs()
    
    print(f"[INFO] Total experiments to run: {len(configs)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Experiment outputs will be saved to {args.output_dir}")
    
    # Run all experiments
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[INFO] Experiment {i}/{len(configs)}")
        success = run_experiment(config, args.train_data, args.test_data, args.output_dir, args)
        results.append(success)
    
    # Create summary report
    summary = create_summary_report(args.output_dir, configs, results)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"{'='*80}\n")
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    exit(main())
