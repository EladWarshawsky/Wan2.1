import logging
import math
import torch
import torch.nn as nn
from safetensors.torch import load_file

class LoRALayer(nn.Module):
    """A wrapper for a linear layer that adds a LoRA path."""
    def __init__(self, linear_layer, rank, lora_alpha):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.lora_alpha = lora_alpha

        self.lora_down = nn.Linear(self.linear.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, self.linear.out_features, bias=False)
        
        # Initialize weights to be zero so it's an identity transformation initially
        nn.init.zeros_(self.lora_up.weight)
        # Kaiming init for lora_down is reasonable even for inference-only application
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))

    def forward(self, x):
        # The scaling factor is alpha / rank
        scale = self.lora_alpha / self.rank
        # Original path + LoRA path
        return self.linear(x) + self.lora_up(self.lora_down(x)) * scale

def load_lora_weights(model, lora_path, alpha=1.0):
    logging.info(f"Loading LoRA weights from: {lora_path}")
    lora_state_dict = load_file(lora_path, device="cpu")

    def normalize_module_path_from_key(key: str):
        """
        Extract module path from LoRA key supporting formats like:
        - diffusion_model.blocks.0.self_attn.q.lora_down.weight
        - model.diffusion_model.blocks.0.cross_attn.k.lora_up.weight
        - lora_unet_blocks_15_self_attn_q.lora_down.weight
        Returns normalized module path like 'blocks.0.self_attn.q', or None if not a LoRA weight.
        """
        if key.endswith(".lora_down.weight"):
            stem = key[: -len(".lora_down.weight")]
            weight_kind = "lora_down.weight"
        elif key.endswith(".lora_up.weight"):
            stem = key[: -len(".lora_up.weight")]
            weight_kind = "lora_up.weight"
        else:
            return None, None

        # Strip known prefixes
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break

        # Handle comfy-style prefix
        if stem.startswith("lora_unet_"):
            # Convert lora_unet_blocks_0_self_attn_q -> blocks.0.self_attn.q
            rest = stem[len("lora_unet_"):]
            stem = rest.replace("_", ".")

        # Small normalizations: sometimes names use 'self.attn' vs 'self_attn'
        stem = stem.replace("self.attn", "self_attn").replace("cross.attn", "cross_attn")
        return stem, weight_kind

    # Build mapping from module path -> {lora_down.weight, lora_up.weight}
    lora_layers = {}
    for key, value in lora_state_dict.items():
        module_path, weight_kind = normalize_module_path_from_key(key)
        if module_path is None:
            continue
        entry = lora_layers.setdefault(module_path, {})
        entry[weight_kind] = value

    logging.info(f"Found {len(lora_layers)} LoRA-modified layers in the file.")
    
    # Inject these layers into the model
    name_set = {n for n, _ in model.named_modules()}
    applied, skipped = 0, 0
    for target_path, weights in lora_layers.items():
        # prefer exact match
        candidate = target_path if target_path in name_set else None
        if candidate is None:
            # try simple variants: sometimes stems already match; we can also try replacing any accidental '..'
            alt = target_path.replace("..", ".")
            if alt in name_set:
                candidate = alt

        if candidate is None:
            skipped += 1
            logging.debug(f"LoRA target not found in model: {target_path}")
            continue

        module = model.get_submodule(candidate)
        if not isinstance(module, nn.Linear):
            skipped += 1
            logging.debug(f"LoRA target {candidate} exists but is {type(module).__name__}, expected nn.Linear. Skipping.")
            continue

        lora_down_weight = weights.get("lora_down.weight")
        lora_up_weight = weights.get("lora_up.weight")
        if lora_down_weight is None or lora_up_weight is None:
            skipped += 1
            logging.debug(f"Incomplete LoRA weights for {candidate}. Skipping.")
            continue

        rank = lora_down_weight.shape[0]
        lora_layer = LoRALayer(module, rank, alpha)
        lora_layer.lora_down.weight.data = lora_down_weight.to(device=module.weight.device, dtype=module.weight.dtype)
        lora_layer.lora_up.weight.data = lora_up_weight.to(device=module.weight.device, dtype=module.weight.dtype)

        parent_module = model.get_submodule(".".join(candidate.split('.')[:-1]))
        child_name = candidate.split('.')[-1]
        setattr(parent_module, child_name, lora_layer)
        applied += 1
        logging.info(f"Patched LoRA into: {candidate} (rank={rank})")

    logging.info(f"LoRA patch summary: applied={applied}, skipped={skipped}, total_targets={len(lora_layers)}")

    return model
