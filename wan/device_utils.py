import torch

def get_optimal_device():
    """
    Returns the optimal device available on the system.
    Priority: CUDA > MPS (for Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def get_current_device():
    """
    Returns the current device being used by PyTorch.
    If no device is set, returns the optimal device.
    """
    if torch.cuda.is_initialized():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
