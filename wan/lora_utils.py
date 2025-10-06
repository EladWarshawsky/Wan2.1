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

    lora_layers = {}
    # Parse keys for common LoRA formats (e.g., ComfyUI/Diffusers-like)
    for key, value in lora_state_dict.items():
        # Example key: "lora_unet_blocks_15_self_attn_q.lora_down.weight"
        parts = key.split('.')
        if len(parts) < 4 or "lora" not in parts[0]:
            continue

        # Reconstruct the target module name
        # e.g., "blocks.15.self_attn.q"
        module_path = parts[0].replace("lora_unet_", "").replace("_", ".")
        
        # Store the lora_down and lora_up weights for this module
        if module_path not in lora_layers:
            lora_layers[module_path] = {}
        
        # The weight type is 'lora_down.weight' or 'lora_up.weight'
        weight_type = ".".join(parts[1:])
        lora_layers[module_path][weight_type] = value

    logging.info(f"Found {len(lora_layers)} LoRA-modified layers in the file.")
    
    # Inject these layers into the model
    for name, module in model.named_modules():
        if name in lora_layers:
            if not isinstance(module, nn.Linear):
                logging.warning(f"LoRA target {name} is not a nn.Linear layer. Skipping.")
                continue
            
            lora_data = lora_layers[name]
            lora_down_weight = lora_data.get("lora_down.weight")
            lora_up_weight = lora_data.get("lora_up.weight")

            if lora_down_weight is None or lora_up_weight is None:
                logging.warning(f"Incomplete LoRA weights for {name}. Skipping.")
                continue

            rank = lora_down_weight.shape[0]
            
            # Create and patch the new LoRA layer
            lora_layer = LoRALayer(module, rank, alpha)
            lora_layer.lora_down.weight.data = lora_down_weight.to(device=module.weight.device, dtype=module.weight.dtype)
            lora_layer.lora_up.weight.data = lora_up_weight.to(device=module.weight.device, dtype=module.weight.dtype)
            
            # Replace the original linear layer with our new LoRA-wrapped layer
            parent_module = model.get_submodule(".".join(name.split('.')[:-1]))
            child_name = name.split('.')[-1]
            setattr(parent_module, child_name, lora_layer)
            logging.info(f"Successfully patched LoRA into: {name}")

    return model
