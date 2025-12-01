import numpy as np
import torch

from map2map.models import UNet
from gaussian_npe import gaussian_npe_model, utils

import asyncio
from time import time
from datetime import datetime
import os

from typing import Callable, Tuple, Optional

def chebyshev_coefficients_log(n: int, a: float, b: float) -> torch.Tensor:
    """
    Compute Chebyshev coefficients for log(x) on interval [a, b].
    
    Args:
        n: Number of Chebyshev coefficients
        a: Lower bound of interval
        b: Upper bound of interval
    
    Returns:
        Tensor of Chebyshev coefficients
    """
    # Chebyshev nodes in [-1, 1]
    k = torch.arange(n, dtype=torch.float64)
    nodes = torch.cos((2*k + 1) * np.pi / (2*n))
    
    # Map to [a, b]
    x = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    
    # Function values at Chebyshev nodes
    f_vals = torch.log(x)
    
    # Compute Chebyshev coefficients using DCT
    coeffs = torch.zeros(n, dtype=torch.float64)
    for j in range(n):
        cos_vals = torch.cos(j * torch.arccos(nodes))
        coeffs[j] = torch.sum(f_vals * cos_vals) * (2.0 / n)
    
    coeffs[0] /= 2.0  # Adjust first coefficient
    
    return coeffs

def evaluate_chebyshev_matvec(Q_op: Callable[[torch.Tensor], torch.Tensor], 
                             v: torch.Tensor, 
                             coeffs: torch.Tensor, 
                             a: float, 
                             b: float,
                             vector_shape: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    """
    Evaluate Chebyshev polynomial approximation of log(Q) applied to vector v.
    
    Args:
        Q_op: Function that applies Q to a vector
        v: Input vector (flattened if vector_shape is provided)
        coeffs: Chebyshev coefficients
        a: Lower bound of spectrum
        b: Upper bound of spectrum
        vector_shape: Original shape of vectors that Q_op expects (e.g., (batch_size, 3))
    
    Returns:
        Approximation of log(Q) @ v
    """
    n = len(coeffs)
    device = v.device
    dtype = v.dtype
    
    # Map Q to [-1, 1]: T = (2*Q - (a+b)*I) / (b-a)
    def T_op(x):
        if vector_shape is not None:
            # Reshape to expected format, apply Q, then flatten
            x_reshaped = x.view(vector_shape)
            result = Q_op(x_reshaped)
            return (2.0 * result.flatten() - (a + b) * x) / (b - a)
        else:
            return (2.0 * Q_op(x) - (a + b) * x) / (b - a)
    
    # Initialize Chebyshev recurrence
    if n == 0:
        return torch.zeros_like(v)
    
    T_prev_prev = torch.zeros_like(v)  # T_0(T) @ v = v
    T_prev = v.clone()  # T_0(T) @ v = v
    
    result = coeffs[0].to(device=device, dtype=dtype) * v
    
    if n > 1:
        T_curr = T_op(v)  # T_1(T) @ v = T @ v
        result += coeffs[1].to(device=device, dtype=dtype) * T_curr
        
        # Chebyshev recurrence: T_{n+1} = 2*T*T_n - T_{n-1}
        for k in range(2, n):
            T_next = 2.0 * T_op(T_curr) - T_prev
            result += coeffs[k].to(device=device, dtype=dtype) * T_next
            
            T_prev = T_curr
            T_curr = T_next
    
    return result

def estimate_spectrum_bounds(Q_op: Callable[[torch.Tensor], torch.Tensor], 
                           vector_shape: Tuple[int, ...], 
                           num_iter: int = 50, 
                           device: str = 'cuda') -> Tuple[float, float]:
    """
    Estimate spectral bounds of Q using power iteration.
    
    Args:
        Q_op: Function that applies Q to a vector
        vector_shape: Shape of vectors that Q_op expects (e.g., (batch_size, 3))
        num_iter: Number of power iterations
        device: Device for computations
    
    Returns:
        Tuple of (lambda_min, lambda_max) estimates
    """
    # Total dimension
    n = int(np.prod(vector_shape))
    
    # Estimate largest eigenvalue
    v = torch.randn(vector_shape, device=device)
    v = v / torch.norm(v)
    
    for _ in range(num_iter):
        v = Q_op(v)
        v = v / torch.norm(v)
    
    lambda_max = torch.sum(v * Q_op(v)).item()
    
    # Estimate smallest eigenvalue using inverse iteration
    # Approximate using Q - shift*I where shift is slightly less than lambda_max
    shift = 0.9 * lambda_max
    
    def shifted_Q_op(x):
        return Q_op(x) - shift * x
    
    v = torch.randn(vector_shape, device=device)
    v = v / torch.norm(v)
    
    # Simple approximation - in practice you'd solve (Q - shift*I)^{-1} v
    # Here we use a few steps of gradient descent as a rough approximation
    for _ in range(20):
        grad = shifted_Q_op(v)
        v = v - 0.1 * grad
        v = v / torch.norm(v)
    
    lambda_min = torch.sum(v * Q_op(v)).item()
    lambda_min = max(lambda_min, 1e-8)  # Ensure positive
    
    # Add some safety margin
    lambda_min *= 0.9
    lambda_max *= 1.1
    
    return lambda_min, lambda_max

def stochastic_logdet_chebyshev(Q_op: Callable[[torch.Tensor], torch.Tensor],
                              vector_shape: Tuple[int, ...],
                              num_samples: int = 50,
                              num_chebyshev: int = 50,
                              spectrum_bounds: Optional[Tuple[float, float]] = None,
                              device: str = 'cuda',
                              seed: Optional[int] = None) -> float:
    """
    Estimate log det(Q) using Chebyshev approximation and stochastic trace estimation.
    
    Args:
        Q_op: Function that applies matrix Q to a vector with shape vector_shape
        vector_shape: Shape of vectors that Q_op expects (e.g., (batch_size, 3) for batches of 3D vectors)
        num_samples: Number of random vectors for trace estimation
        num_chebyshev: Number of Chebyshev coefficients
        spectrum_bounds: Optional tuple (lambda_min, lambda_max). If None, will estimate.
        device: Device for computations
        seed: Random seed
    
    Returns:
        Estimate of log det(Q)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Total dimension
    n = int(np.prod(vector_shape))
    
    # Estimate spectrum bounds if not provided
    if spectrum_bounds is None:
        print("Estimating spectrum bounds...")
        lambda_min, lambda_max = estimate_spectrum_bounds(Q_op, vector_shape, device=device)
        print(f"Estimated spectrum: [{lambda_min:.6f}, {lambda_max:.6f}]")
    else:
        lambda_min, lambda_max = spectrum_bounds
    
    # Compute Chebyshev coefficients for log(x) on [lambda_min, lambda_max]
    print(f"Computing {num_chebyshev} Chebyshev coefficients...")
    coeffs = chebyshev_coefficients_log(num_chebyshev, lambda_min, lambda_max)
    
    # Stochastic trace estimation
    print(f"Running stochastic trace estimation with {num_samples} samples...")
    trace_estimates = []
    
    for i in range(num_samples):
        # Generate random Rademacher vector in flattened form
        z_flat = torch.randint(0, 2, (n,), device=device, dtype=torch.float32) * 2 - 1
        
        # Compute log(Q) @ z using Chebyshev approximation
        log_Q_z_flat = evaluate_chebyshev_matvec(Q_op, z_flat, coeffs, lambda_min, lambda_max, vector_shape)
        
        # Compute trace contribution: z^T log(Q) z
        trace_contribution = torch.dot(z_flat, log_Q_z_flat).item()
        trace_estimates.append(trace_contribution)
        
        if (i + 1) % 10 == 0:
            current_estimate = np.mean(trace_estimates)
            std_estimate = np.std(trace_estimates) / np.sqrt(i + 1)
            print(f"Sample {i+1:3d}: current estimate = {current_estimate:.6f} ± {std_estimate:.6f}")
    
    logdet_estimate = np.mean(trace_estimates)
    std_error = np.std(trace_estimates) / np.sqrt(num_samples)
    
    print(f"\nFinal estimate: log det(Q) = {logdet_estimate:.6f} ± {std_error:.6f}")
    
    return logdet_estimate

def conjugate_gradient_batched(A_fn, b, x0=None, M_inv=None, tol=1e-10, max_iter=10_000):
    """
    Batched Conjugate Gradient solver with optional diagonal preconditioner.

    Args:
        A_fn: Function that computes Ax, applied element-wise to batches.
        b: Right-hand side tensor of shape (B, *).
        x0: Optional initial guess of same shape.
        M_inv: Optional preconditioner tensor of same shape.
        tol: Tolerance for convergence (scalar or per-batch).
        max_iter: Maximum number of iterations.

    Returns:
        x: Solution tensor of same shape as b.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_fn(x)
    z = r if M_inv is None else M_inv * r
    p = z.clone()
    rz_old = torch.sum(r * z, dim=tuple(range(1, r.ndim)), keepdim=True)  # Shape (B, 1, ..., 1)

    for _ in range(max_iter):
        Ap = A_fn(p)
        pAp = torch.sum(p * Ap, dim=tuple(range(1, p.ndim)), keepdim=True)
        alpha = rz_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        r_norm = torch.norm(r.view(r.shape[0], -1), dim=1)  # shape (B,)
        if torch.all(r_norm < tol):
            print(f"Converged after {_+1} iterations")
            break
        z = r if M_inv is None else M_inv * r
        rz_new = torch.sum(r * z, dim=tuple(range(1, r.ndim)), keepdim=True)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
    return x

def stochastic_logdet_gradient(apply_Q, shape, num_samples=8):
    """
    Approximates gradient of log det(Q) using stochastic trace estimation.
    
    apply_Q: function(v) that computes Q(theta) @ v
    theta: parameters (require gradients)
    dim: dimension of Q
    num_samples: number of probe vectors
    """
    Z = torch.randn(num_samples, *shape, device='cuda')  # z_i
    # Z.requires_grad = False  # make sure it's not tracked

    with torch.no_grad():
        X = conjugate_gradient_batched(apply_Q, Z).detach()  # Q^{-1} z, detached
        # print("X", X.shape)

    # Construct trace estimator with dQ/dθ z tracked
    # trace_estimate = torch.sum(Z * X) / num_samples
    X = X.detach().requires_grad_(False)
    trace_estimate = (X * apply_Q(Z)).mean() #.sum(dim=(-3, -2, -1))
    return trace_estimate #.mean()

def trace_estimation(apply_Q, shape, num_samples=16):
    """
    Approximates gradient of log det(Q) using stochastic trace estimation.
    
    apply_Q: function(v) that computes Q(theta) @ v
    theta: parameters (require gradients)
    dim: dimension of Q
    num_samples: number of probe vectors
    """
    Z = torch.randn(num_samples, *shape, device='cuda')  # z_i
    # Z.requires_grad = False  # make sure it's not tracked

    trace_estimate = (Z * apply_Q(Z)).mean()  #mean over Z vectors and pixels
    return trace_estimate
