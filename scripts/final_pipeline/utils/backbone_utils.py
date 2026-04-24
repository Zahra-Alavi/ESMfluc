#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import types
from pathlib import Path

import huggingface_hub
import torch
from transformers import EsmModel, EsmTokenizer

from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

# Compatibility patch for EsmSequenceTokenizer with newer transformers.
_ESM3_SPECIAL_TOKENS = {
    "cls_token": "<cls>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "eos_token": "<eos>",
    "bos_token": "<cls>",
    "unk_token": "<unk>",
}


def _esm_get_token(self, token_name: str) -> str:
    if token_name in _ESM3_SPECIAL_TOKENS:
        return _ESM3_SPECIAL_TOKENS[token_name]
    raise AttributeError(f"Unknown special token name: {token_name!r}")


def _noop_setter(self, value):
    pass


for _attr in ("cls_token", "eos_token", "mask_token", "pad_token", "bos_token"):
    _prop = getattr(EsmSequenceTokenizer, _attr, None)
    if isinstance(_prop, property) and _prop.fset is None:
        setattr(EsmSequenceTokenizer, _attr, _prop.setter(_noop_setter))

EsmSequenceTokenizer._get_token = _esm_get_token


def load_hf_token_from_env(env_file: Path | None = None) -> None:
    """Load HF token from local .env (if any) and login once."""
    _env_file = env_file or (Path(__file__).parent / ".env")
    if _env_file.exists():
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

    _hf_token = os.environ.get("HF_TOKEN")
    if _hf_token:
        huggingface_hub.login(token=_hf_token, add_to_git_credential=False)


def is_esm3_name(model_name: str) -> bool:
    return "esm3" in model_name.lower()


def build_tokenizer(esm_model_name: str):
    if is_esm3_name(esm_model_name):
        return EsmSequenceTokenizer()
    return EsmTokenizer.from_pretrained(f"facebook/{esm_model_name}")


def tokenize(sequences, tokenizer):
    out = []
    for seq in sequences:
        enc = tokenizer(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        if "attention_mask" not in enc:
            pad_id = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 1
            enc["attention_mask"] = (enc["input_ids"] != pad_id).long()
        out.append(enc)
    return out


def set_up_embedding_model(args):
    args.is_esm3 = is_esm3_name(args.esm_model)

    if args.is_esm3:
        embedding_model = ESM3_sm_open_v0(str(args.device))
        embedding_model.is_esm3 = True

        if not getattr(args, "freeze_all_backbone", False) and not args.freeze_layers:
            print(
                "Warning: ESM3 is fully trainable with current settings. "
                "This is memory-intensive and often OOMs at long sequence lengths. "
                "Try --freeze_all_backbone or --freeze_layers with --mixed_precision and smaller --batch_size."
            )

        # Expose hidden_size so existing model constructors still work
        embedding_model.config = types.SimpleNamespace(hidden_size=1536)

        if getattr(args, "freeze_all_backbone", False):
            for p in embedding_model.parameters():
                p.requires_grad = False
            embedding_model.eval()
            print("Frozen ALL ESM3 parameters.")
            return embedding_model

        embedding_model.train()

        if args.freeze_layers:
            start_layer, end_layer = map(int, args.freeze_layers.split("-"))
            freeze_set = set(range(start_layer, end_layer + 1))
            for idx, block in enumerate(embedding_model.transformer.blocks):
                req = idx not in freeze_set
                for p in block.parameters():
                    p.requires_grad = req
            print(f"Freezing ESM3 layers {args.freeze_layers}")

        n_trainable = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
        print(f"Trainable ESM3 params (elements): {n_trainable:,}")
        return embedding_model

    # ESM2 path
    embedding_model = EsmModel.from_pretrained(f"facebook/{args.esm_model}")
    embedding_model.is_esm3 = False
    embedding_model.to(args.device)

    if getattr(args, "freeze_all_backbone", False):
        for p in embedding_model.parameters():
            p.requires_grad = False
        embedding_model.eval()
        print("Frozen ALL ESM parameters.")
        n_trainable = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
        print(f"Trainable ESM params (elements): {n_trainable:,}")
        return embedding_model

    embedding_model.train()

    if args.freeze_layers:
        start_layer, end_layer = map(int, args.freeze_layers.split("-"))
        freeze_list = range(start_layer, end_layer + 1)
        for name, param in embedding_model.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split(".")[2])
                param.requires_grad = layer_num not in freeze_list
            else:
                param.requires_grad = True
        print(f"Freezing layers {args.freeze_layers}")

    n_trainable = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
    print(f"Trainable ESM params (elements): {n_trainable:,}")
    return embedding_model
