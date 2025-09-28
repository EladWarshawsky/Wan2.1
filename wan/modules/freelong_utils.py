# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Adapted for FreeLong++ implementation.

import torch
import torch.fft as fft


def gaussian_low_pass_filter(shape, d_s=0.25, d_t=0.25):
    """
    Computes a 3D Gaussian low-pass filter mask.
    Args:
        shape: Shape of the filter (T, H, W).
        d_s: Normalized stop frequency for spatial dimensions (0.0-1.0).
        d_t: Normalized stop frequency for temporal dimension (0.0-1.0).
    """
    T, H, W = shape[-3], shape[-2], shape[-1]
    if d_s <= 0 or d_t <= 0:
        return torch.zeros(shape)

    # Create normalized coordinate grids for T, H, W
    t = (torch.arange(T).float() / T - 0.5) * 2
    h = (torch.arange(H).float() / H - 0.5) * 2
    w = (torch.arange(W).float() / W - 0.5) * 2
    
    grid_t, grid_h, grid_w = torch.meshgrid(t, h, w, indexing='ij')

    # Compute squared distance from the center, adjusted for frequency cutoffs
    # The cutoff frequencies d_s and d_t correspond to the standard deviation of the Gaussian
    d_square = (grid_t / d_t).pow(2) + (grid_h / d_s).pow(2) + (grid_w / d_s).pow(2)
    
    # Compute the Gaussian mask
    mask = torch.exp(-0.5 * d_square)
    
    return mask


def freq_mix_3d(x, noise, LPF):
    """
    Performs frequency mixing of two signals using a Low-Pass Filter.
    This is the core function for the original FreeLong's dual-branch fusion.
    Args:
        x: The signal for low-frequency components.
        noise: The signal for high-frequency components.
        LPF: The low-pass filter mask.
    """
    # FFT
    x_freq = fft.fftshift(fft.fftn(x, dim=(-3, -2, -1)), dim=(-3, -2, -1))
    noise_freq = fft.fftshift(fft.fftn(noise, dim=(-3, -2, -1)), dim=(-3, -2, -1))

    # Frequency mix
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    x_freq_mixed = x_freq_low + noise_freq_high

    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-3, -2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-3, -2, -1)).real

    return x_mixed


def create_band_pass_filters(shape, alphas, d_s=0.25):
    """
    Creates a set of mutually exclusive band-pass filters for FreeLong++.
    Args:
        shape: The shape of the filters (T, H, W).
        alphas: A list of window scale factors, e.g., [1, 2, 4, "global"].
                The largest numeric alpha corresponds to the lowest frequency band.
        d_s: Normalized spatial stop frequency.
    """
    filters = {}
    # Sort numeric alphas from largest to smallest (coarsest to finest temporal window)
    sorted_numeric_alphas = sorted([a for a in alphas if isinstance(a, int)], reverse=True)

    # 1. Start with the lowest frequency band (from the largest window, e.g., alpha=4)
    # It is a simple Low-Pass Filter. Cutoff frequency is inversely proportional to window size.
    d_t_coarsest = 0.25 / sorted_numeric_alphas[0]
    prev_lpf = gaussian_low_pass_filter(shape, d_s, d_t_coarsest)
    filters[sorted_numeric_alphas[0]] = prev_lpf

    # 2. Create intermediate band-pass filters by subtracting LPFs
    for i in range(len(sorted_numeric_alphas) - 1):
        alpha_curr = sorted_numeric_alphas[i+1]
        d_t_curr = 0.25 / alpha_curr
        current_lpf = gaussian_low_pass_filter(shape, d_s, d_t_curr)
        
        # A band-pass filter is the region between two low-pass filters
        band_pass = current_lpf - prev_lpf
        filters[alpha_curr] = band_pass
        prev_lpf = current_lpf
        
    # 3. The highest frequency band is everything not covered by the last LPF
    # This corresponds to the smallest window size (alpha=1).
    filters[sorted_numeric_alphas[-1]] = 1.0 - prev_lpf

    # 4. Map the "global" key to the lowest frequency filter for convenience
    if "global" in alphas:
        filters["global"] = filters[sorted_numeric_alphas[0]]
        
    return filters
