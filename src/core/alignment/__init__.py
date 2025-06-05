"""Monotonic alignment search module."""

import numpy as np
import torch
from typing import Optional

try:
    from .core import maximum_path_c
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Warning: Cython monotonic alignment not available, using fallback implementation")


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Maximum path search for monotonic alignment.
    
    Args:
        neg_cent: Negative centroids [batch, text_len, spec_len]
        mask: Attention mask [batch, text_len, spec_len]
        
    Returns:
        Alignment path [batch, text_len, spec_len]
    """
    if CYTHON_AVAILABLE:
        return _maximum_path_cython(neg_cent, mask)
    else:
        return _maximum_path_fallback(neg_cent, mask)


def _maximum_path_cython(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cython optimized version."""
    # For now, fall back to the Python implementation
    return _maximum_path_fallback(neg_cent, mask)


def _maximum_path_fallback(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fallback implementation in pure Python/PyTorch."""
    device = neg_cent.device
    dtype = neg_cent.dtype
    b, t_t, t_s = neg_cent.shape
    
    # Convert to numpy for processing
    neg_cent_np = neg_cent.detach().cpu().numpy()
    path = np.zeros((b, t_t, t_s), dtype=np.float32)
    
    for batch_idx in range(b):
        path[batch_idx] = _maximum_path_single(
            neg_cent_np[batch_idx]
        )
    
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def _maximum_path_single(neg_cent: np.ndarray) -> np.ndarray:
    """Single batch maximum path search for monotonic alignment."""
    t_t, t_s = neg_cent.shape
    path = np.zeros((t_t, t_s), dtype=np.float32)
    
    # Dynamic programming matrix
    Q = np.full((t_t, t_s), -np.inf, dtype=np.float32)
    
    # Initialize
    Q[0, 0] = neg_cent[0, 0]
    
    # Fill first row (can only move right)
    for j in range(1, t_s):
        Q[0, j] = Q[0, j-1] + neg_cent[0, j]
    
    # Fill first column (can only move down)
    for i in range(1, t_t):
        Q[i, 0] = Q[i-1, 0] + neg_cent[i, 0]
    
    # Fill the rest of the matrix
    for i in range(1, t_t):
        for j in range(1, t_s):
            # Can come from left, top, or diagonal (top-left)
            Q[i, j] = max(Q[i-1, j], Q[i, j-1], Q[i-1, j-1]) + neg_cent[i, j]
    
    # Backtrack from bottom-right to find the optimal path
    i, j = t_t - 1, t_s - 1
    
    while i >= 0 and j >= 0:
        path[i, j] = 1
        
        if i == 0 and j == 0:
            break
        elif i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # Choose the direction that led to the maximum
            if Q[i-1, j-1] >= Q[i-1, j] and Q[i-1, j-1] >= Q[i, j-1]:
                i -= 1
                j -= 1
            elif Q[i-1, j] >= Q[i, j-1]:
                i -= 1
            else:
                j -= 1
    
    return path
