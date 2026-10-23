"""Training arguments for the seven published binary model conditions."""
import argparse
from pathlib import Path

DEFAULT_SPLIT_DIR = Path(__file__).resolve().parents[2] / "data" / "atlas_grouped_v1"
CONDITIONS = {
    "esm2_frozen_linear": ("esm2_t33_650M_UR50D", "esm_linear", True, None),
    "esm2_frozen_bilstm_attn": ("esm2_t33_650M_UR50D", "bilstm_attention", True, None),
    "esm2_top4_bilstm_attn": ("esm2_t33_650M_UR50D", "bilstm_attention", False, "0-28"),
    "esm2_top28_bilstm_attn": ("esm2_t33_650M_UR50D", "bilstm_attention", False, "0-4"),
    "esm3_frozen_bilstm_attn": ("esm3_sm_open_v1", "bilstm_attention", True, None),
    "esm3_top4_bilstm_attn": ("esm3_sm_open_v1", "bilstm_attention", False, "0-43"),
    "esm3_top28_bilstm_attn": ("esm3_sm_open_v1", "bilstm_attention", False, "0-19"),
}


class TrainingParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        values = super().parse_args(args, namespace)
        names = ("esm_model", "architecture", "freeze_all_backbone", "freeze_layers")
        if values.condition:
            for name, expected in zip(names, CONDITIONS[values.condition]):
                given = getattr(values, name)
                if given is not None and given != expected:
                    self.error(f"--{name} conflicts with --condition {values.condition}")
                setattr(values, name, expected)
        else:
            values.esm_model = values.esm_model or "esm2_t33_650M_UR50D"
            values.architecture = values.architecture or "bilstm_attention"
            if values.freeze_all_backbone is None:
                values.freeze_all_backbone = values.freeze_layers is None
            configuration = tuple(getattr(values, name) for name in names)
            values.condition = next((key for key, value in CONDITIONS.items()
                                     if value == configuration), None)
            if values.condition is None:
                self.error("Use one of the seven published --condition settings.")
        if not 1 <= values.epochs <= 80:
            self.error("--epochs must be between 1 and 80")
        values.data_parallel = False
        values.focal_class_weights = False
        return values


def parse_arguments():
    parser = TrainingParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS)
    for split in ("train", "validation", "test"):
        parser.add_argument(f"--{split}_data_file",
                            default=str(DEFAULT_SPLIT_DIR / f"{split}_grouped_v1.csv"))
    parser.add_argument("--esm_model", choices=["esm2_t33_650M_UR50D", "esm3_sm_open_v1"])
    parser.add_argument("--architecture", choices=["bilstm_attention", "esm_linear"])
    parser.add_argument("--freeze_all_backbone", action="store_true", default=None)
    parser.add_argument("--freeze_layers", choices=["0-28", "0-4", "0-43", "0-19"])
    for name, value in (("batch_size", 2), ("gradient_accumulation_steps", 1),
                        ("patience", 5), ("hidden_size", 512), ("num_layers", 3),
                        ("bidirectional", 1), ("num_classes", 2)):
        parser.add_argument(f"--{name}", type=int, default=value, choices=[value])
    for name, value in (("lr", 1e-5), ("weight_decay", 1e-2), ("dropout", 0.3)):
        parser.add_argument(f"--{name}", type=float, default=value, choices=[value])
    parser.add_argument("--epochs", type=int, default=80,
                        help="At most 80; fewer epochs may be used for a smoke test.")
    parser.add_argument("--neq_thresholds", type=float, nargs=1, choices=[1.0], default=[1.0])
    parser.add_argument("--loss_function", choices=["focal"], default="focal")
    parser.add_argument("--lr_scheduler", choices=["reduce_on_plateau"], default="reduce_on_plateau")
    parser.add_argument("--task_type", choices=["classification"], default="classification")
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp_dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--result_foldername", default="timestamp")
    return parser
