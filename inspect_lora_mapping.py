#!/usr/bin/env python3
import argparse
import os
import logging
from collections import Counter

import torch
from safetensors.torch import load_file

import wan
from wan.configs import WAN_CONFIGS

def main():
    parser = argparse.ArgumentParser("Inspect LoRA ↔ VACE model mapping")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--task", type=str, default="vace-1.3B", help="Task (default: vace-1.3B)")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA .safetensors file")
    parser.add_argument("--print_modules", type=int, default=200, help="How many module names to print")
    parser.add_argument("--print_lora_keys", type=int, default=50, help="How many LoRA keys to print")
    parser.add_argument("--suffix_depth", type=int, default=3, help="Suffix depth for fuzzy matching")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    # 1) Check which load path will be used
    base_t2v_path = os.path.join(args.ckpt_dir, "wan2.1_t2v_1.3B_fp16.safetensors")
    vace_weights_path = os.path.join(args.ckpt_dir, "diffusion_pytorch_model.safetensors")
    logging.info(f"ckpt_dir: {args.ckpt_dir}")
    logging.info(f"Exists base T2V? {os.path.exists(base_t2v_path)} -> {base_t2v_path}")
    logging.info(f"Exists VACE weights? {os.path.exists(vace_weights_path)} -> {vace_weights_path}")
    if os.path.exists(base_t2v_path) and os.path.exists(vace_weights_path):
        logging.info("Expected load path: base T2V weights + VACE-specific weights")
    else:
        logging.info("Expected load path: VACE standard checkpoint only (module names may differ from base T2V)")

    # 2) Build config and load model similarly to generate.py
    cfg = WAN_CONFIGS[args.task]
    logging.info(f"Using config for {args.task}: {cfg.__name__ if hasattr(cfg,'__name__') else 'dict'}")
    # device: keep CPU to avoid VRAM, we only need names
    wan_vace = wan.WanVace(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=torch.device("cpu"),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=True,
        disable_token_budget=False,
    )

    # 3) Collect model module names and types
    names = []
    types = []
    for name, mod in wan_vace.model.named_modules():
        names.append(name)
        types.append(type(mod).__name__)
    logging.info(f"Total modules: {len(names)}")
    type_counts = Counter(types)
    logging.info(f"Module type counts (top 10): {type_counts.most_common(10)}")

    # Print first N modules
    print("\n=== First modules ===")
    for i, (n, t) in enumerate(zip(names, types)):
        if i >= args.print_modules:
            break
        print(f"{i:04d}: {n} [{t}]")

    # Heuristic: show candidates for attention projections
    print("\n=== Likely q/k/v projection modules (first 100) ===")
    hits = 0
    for n, t in zip(names, types):
        if any(tok in n.lower() for tok in ["to_q", "to_k", "to_v", "q_proj", "k_proj", "v_proj", ".q", ".k", ".v"]):
            print(f"{n} [{t}]")
            hits += 1
            if hits >= 100:
                break
    if hits == 0:
        print("(no obvious q/k/v naming hits; projections may be named differently)")

    # 4) Load LoRA keys
    logging.info(f"Loading LoRA keys from: {args.lora_path}")
    lora_sd = load_file(args.lora_path, device="cpu")
    lora_keys = list(lora_sd.keys())
    logging.info(f"Total LoRA keys: {len(lora_keys)}")
    print("\n=== First LoRA keys ===")
    for i, k in enumerate(lora_keys[:args.print_lora_keys]):
        print(f"{i:04d}: {k}")

    # 5) Try to derive target module paths from LoRA keys (several schemes)
    def derive_module_path(key: str):
        # Supports patterns like:
        # - lora_unet_blocks_15_self_attn_q.lora_down.weight
        # - blocks.15.attn.to_q.lora_A / lora_B
        # - arbitrary prefix ... we’ll try to find the segment before lora_down/up or lora_A/B
        parts = key.split(".")
        # find index of lora token
        lora_idx = None
        for idx, p in enumerate(parts):
            if p in ("lora_down", "lora_up", "lora_A", "lora_B"):
                lora_idx = idx
                break
        if lora_idx is None:
            return None

        prefix = ".".join(parts[:lora_idx])  # e.g., 'lora_unet_blocks_15_self_attn_q' OR 'blocks.15.self_attn.q'
        # normalize common prefixes
        if prefix.startswith("lora_unet_"):
            prefix = prefix[len("lora_unet_"):]
        # Some LoRA dumps use underscores where model uses dots
        candidate = prefix.replace("_", ".")
        return candidate

    # Build module set for quick membership tests
    module_set = set(names)

    # Extract candidate module paths from LoRA keys
    candidates = []
    for k in lora_keys:
        mp = derive_module_path(k)
        if mp:
            candidates.append(mp)
    unique_candidates = sorted(set(candidates))

    print("\n=== Derived LoRA module path candidates (first 100) ===")
    for i, c in enumerate(unique_candidates[:100]):
        print(f"{i:04d}: {c}")
    logging.info(f"Total unique candidates derived: {len(unique_candidates)}")

    # 6) Exact matches
    exact_matches = [c for c in unique_candidates if c in module_set]
    missing = [c for c in unique_candidates if c not in module_set]
    logging.info(f"Exact matches: {len(exact_matches)}")
    logging.info(f"Missing (no exact match): {len(missing)}")
    print("\n=== Exact matches (first 100) ===")
    for i, c in enumerate(exact_matches[:100]):
        print(f"{i:04d}: {c}")

    # 7) Suffix fuzzy matches for missing
    def suffix_key(name: str, depth: int):
        parts = name.split(".")
        return ".".join(parts[-depth:]) if len(parts) >= depth else name

    print("\n=== Suffix fuzzy matches for missing (first 50) ===")
    miss_hits = 0
    suffix_map = {}
    # Build reverse index by suffix depth
    for n in names:
        suffix = suffix_key(n, args.suffix_depth)
        suffix_map.setdefault(suffix, []).append(n)

    for m in missing:
        suf = suffix_key(m, args.suffix_depth)
        if suf in suffix_map:
            print(f"Missing candidate: {m}")
            for tgt in suffix_map[suf][:5]:
                print(f"  -> possible match: {tgt}")
            miss_hits += 1
            if miss_hits >= 50:
                break
    if miss_hits == 0 and missing:
        print("(no suffix matches found at this depth; try --suffix_depth 2 or 4)")

    # 8) Layer type summary for candidates
    print("\n=== Candidate target layer types (first 100) ===")
    name_to_type = {n:t for n,t in zip(names, types)}
    cnt = 0
    for c in unique_candidates:
        if c in name_to_type:
            print(f"{c} -> {name_to_type[c]}")
            cnt += 1
            if cnt >= 100: break

    print("\nDone.")

if __name__ == "__main__":
    main()
