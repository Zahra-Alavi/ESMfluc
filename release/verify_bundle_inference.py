"""Compare CPU predictions and exact contributions before/after restoration.

No pretrained weights are downloaded: architecture and tokenizer information
come from the explicitly supplied local cache. Runs one selected checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint_bundle import load_bundle_state, read_manifest, model_entry
from esmfluc.inference.get_attn import (
    ESM3Wrapper, EsmSequenceTokenizer, run_model_bilstm_attn,
)
from esmfluc.training.models import BiLSTMWithSelfAttentionModel, ESMLinearTokenClassifier


def make_model(args: dict, cache: Path):
    if args["esm_model"].startswith("esm3"):
        from esm.models.esm3 import ESM3
        from esm.pretrained import (ESM3_structure_encoder_v0,
                                    ESM3_structure_decoder_v0, ESM3_function_decoder_v0)
        from esm.tokenization import get_esm3_model_tokenizers
        from esm.utils.constants.models import ESM3_OPEN_SMALL
        tokenizers = get_esm3_model_tokenizers(ESM3_OPEN_SMALL)
        # ESM3's nonpersistent rotary buffers must be initialized on a real
        # device; they are deliberately absent from checkpoint state_dicts.
        backbone = ESM3Wrapper(ESM3(
            d_model=1536, n_heads=24, v_heads=256, n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=tokenizers))
        tokenizer = EsmSequenceTokenizer()
    else:
        from transformers import EsmConfig, EsmModel, EsmTokenizer
        hub = cache / "models--facebook--esm2_t33_650M_UR50D"
        revision = (hub / "refs/main").read_text().strip()
        config_dir = hub / "snapshots" / revision
        config = EsmConfig.from_pretrained(config_dir, local_files_only=True)
        with torch.device("meta"):
            backbone = EsmModel(config)
        tokenizer = EsmTokenizer.from_pretrained(config_dir, local_files_only=True)
    with torch.device("meta"):
        if args["architecture"] == "esm_linear":
            model = ESMLinearTokenClassifier(backbone, num_classes=2)
        else:
            model = BiLSTMWithSelfAttentionModel(
                backbone, hidden_size=args["hidden_size"], num_layers=args["num_layers"],
                num_classes=2, dropout=args["dropout"], bidirectional=args["bidirectional"])
    return model, tokenizer


def evaluate(model, tokenizer, sequence: str):
    if hasattr(model, "attention"):
        attn, _, pred, score, probs, contributions = run_model_bilstm_attn(
            model, tokenizer, sequence, "cpu", return_logit_contributions=True)
        return {"attention": attn, "predictions": pred.cpu().numpy(),
                "scores": score.numpy(), "probabilities": probs.numpy(),
                **{k: np.asarray(v) for k, v in contributions.items()}}
    encoded = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    model.eval()
    with torch.no_grad():
        logits, _ = model(encoded["input_ids"], encoded["attention_mask"])
    return {"logits": logits.numpy(), "probabilities": logits.softmax(-1).numpy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--upstream-cache", type=Path, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sequence", default="SVGEGPDTKPTARGQPAAVG")
    parser.add_argument("--fasta", type=Path, help="Optional single-sequence FASTA instead")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.fasta is not None:
        lines = args.fasta.read_text().splitlines()
        if sum(line.startswith(">") for line in lines) != 1:
            raise ValueError("Expected exactly one FASTA record")
        args.sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    torch.set_num_threads(4)
    entry = model_entry(read_manifest(args.bundle), args.condition, args.seed)
    model, tokenizer = make_model(entry["training_args"], args.upstream_cache)
    original = torch.load(args.runs / args.condition / f"seed_{args.seed}/best_model.pth",
                          weights_only=True, mmap=True, map_location="cpu")
    model.load_state_dict(original, strict=True, assign=True)
    expected = evaluate(model, tokenizer, args.sequence)
    del original
    restored = load_bundle_state(args.bundle, args.condition, args.seed)
    model.load_state_dict(restored, strict=True, assign=True)
    del restored
    gc.collect()
    actual = evaluate(model, tokenizer, args.sequence)
    errors = {}
    bitwise_equal = True
    for key in expected:
        if not np.isfinite(expected[key]).all() or not np.isfinite(actual[key]).all():
            raise ValueError(f"Nonfinite inference output: {key}")
        errors[key] = float(np.max(np.abs(expected[key] - actual[key])))
        bitwise_equal &= np.array_equal(expected[key], actual[key], equal_nan=True)
        if not np.allclose(expected[key], actual[key], rtol=1e-5, atol=1e-6, equal_nan=True):
            raise ValueError(f"Restored inference differs in {key}: {errors[key]}")
        if key == "predictions" and not np.array_equal(expected[key], actual[key]):
            raise ValueError("Restored predicted classes differ")
    report = {"condition": args.condition, "seed": args.seed, "sequence": args.sequence,
              "device": "cpu", "strict_state_dict_load": True,
              "inference_bitwise_equal": bool(bitwise_equal),
              "inference_numerically_equivalent": True,
              "relative_tolerance": 1e-5, "absolute_tolerance": 1e-6,
              "maximum_absolute_errors": errors,
              "torch_version": torch.__version__,
              "note": "CPU inference equivalence test; not a Colab/GPU compatibility benchmark."}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
