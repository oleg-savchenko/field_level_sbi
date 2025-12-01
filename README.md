# Field-level simulation-based inference

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository implements **field-level simulation-based inference (SBI)** for cosmological initial conditions from masked and noisy observational data. The method uses a Gaussian Neural Posterior Estimation (NPE) framework combined with U-Net architectures to infer the primordial density field from late-time observations while properly accounting for incomplete sky coverage (masking) and observational noise.

## 🔬 Scientific Background

Field-level inference aims to reconstruct the initial conditions of the Universe from observed galaxy distributions or matter density fields. Unlike traditional summary-statistic-based approaches, field-level methods preserve all information in the data, enabling optimal extraction of cosmological information.

Key features of this implementation:
- **Masked data handling**: Properly accounts for survey geometry and masked regions (e.g., galactic plane, bright star masks)
- **Noise modeling**: Incorporates observational noise in the likelihood model
- **Gaussian posterior approximation**: Efficient inference using precision matrix factorization
- **Scalable to large volumes**: Uses stochastic log-determinant estimation (Chebyshev approximation) for tractable inference

## 📁 Repository Structure

```
.
├── field_level_sbi/                 # 🔑 Main inference modules (core of this project)
│   ├── gaussian_npe_model.py        # Precision matrix parameterizations (GDG factorization)
│   ├── gaussian_npe_training        # Training utilities (Chebyshev log-det, CG solver)
│   ├── training_script_masked_data.py  # Training pipeline and NPE network
│   └── utils.py                     # Power spectrum, plotting, mask generation
│
├── falcon/                          # Training framework and simulation infrastructure
│   ├── __init__.py
│   ├── core/                        # Graph-based training pipeline
│   │   ├── graph.py                 # Graphical model definitions (Node, Graph classes)
│   │   ├── deployed_graph.py        # Deployed graph execution
│   │   ├── zarrstore.py             # Zarr dataset management
│   │   └── utils.py                 # Utility functions and lazy loading
│   └── contrib/                     # Additional normalizing flow implementations
│       └── ...
│
├── plots/                           # Output plots and trained models
│   └── [experiment_folders]/        # Contains trained models (.pt) and visualizations
│
├── os250904-falcon_disco_dj_*.py/ipynb    # Disco-DJ based experiments
├── os251016-falcon_Quijote_*.py/ipynb     # Quijote simulation experiments
└── new_my_gpu_jupyter_job.sh        # SLURM job submission script
```

## 🧮 Core Components (`field_level_sbi/`)

The `field_level_sbi/` folder contains the main implementation of the Gaussian NPE method for field-level inference.

### Precision Matrix Parameterizations (`gaussian_npe_model.py`)

The posterior precision matrix $Q_{\text{post}} = Q_{\text{prior}} + Q_{\text{like}}$ is parameterized using the $G^T D G$ factorization:

- **`Precision_Matrix_FFT`**: Diagonal in Fourier space, suitable for translation-invariant priors
- **`Precision_Matrix_Masked_FFT`**: Handles masked data with position-dependent precision
- **`Precision_Matrix_Real`**: Real-space diagonal precision

### Training Utilities (`gaussian_npe_training`)

- **Conjugate Gradient solver**: Batched CG for solving $Q^{-1}z$ in trace estimation
- **Stochastic log-determinant estimation**: Uses Chebyshev polynomial approximation for $\log\det(Q)$
- **Spectrum bound estimation**: Power iteration for eigenvalue bounds

### Network Architecture (`training_script_masked_data.py`)

- **`Gaussian_NPE_Network`**: Main inference network with:
  - U-Net for MAP estimation ($\mu$ network)
  - U-Net for precision matrix parameters ($Q$ network)
  - OU process sampler for posterior sampling

### Utility Functions (`utils.py`)

- **`Power_Spectrum_Sampler`**: Generates Gaussian random fields with specified power spectrum
- **`get_pk_class`**: Computes linear power spectrum using CLASS Boltzmann solver
- **`create_cone_mask`**: Creates cone-shaped survey masks for simulating partial sky coverage
- **Plotting utilities**: Visualization of samples, power spectra, transfer functions, and cross-correlations

## 🔧 FALCON Framework (`falcon/`)

The `falcon/` folder provides the underlying training infrastructure and simulation framework. It implements a graph-based approach for defining probabilistic models and managing the training pipeline. This code is used as the backbone for orchestrating training and forward simulations, but the core inference methodology resides in `field_level_sbi/`.

## 🚀 Installation

### Dependencies

```bash
# Core dependencies
pip install torch>=2.0
pip install numpy matplotlib

# Cosmology packages
pip install classy              # CLASS Boltzmann solver
pip install Pk_library          # Pylians power spectrum library

# Deep learning
pip install map2map             # U-Net implementation

# Optional: JAX for forward simulations
pip install jax jaxlib
pip install discodj             # Differentiable N-body simulations
```

### Full Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Deep learning framework |
| `numpy` | ≥1.20 | Numerical computations |
| `matplotlib` | ≥3.5 | Visualization |
| `classy` | ≥3.0 | Linear power spectrum |
| `Pk_library` | - | Power spectrum estimation |
| `map2map` | - | U-Net architecture |
| `jax` | ≥0.4 | Forward simulations (optional) |
| `zarr` | ≥2.0 | Dataset storage |

## 📊 Results

### Posterior Sampling Animation

The animation below shows the Ornstein-Uhlenbeck (OU) sampling process for the posterior distribution of the initial density field, starting from a prior sample and converging to samples consistent with the masked observed data:

![Posterior Sampling Animation](plots/animation_250908_163622_OU_sampling_Qx_mask_ic.gif)

*OU sampling of the posterior distribution for the initial conditions, demonstrating convergence from prior samples to posterior samples conditioned on masked late-time observations.*

## 📖 Usage

### Training

```python
from field_level_sbi.training_script_masked_data import Gaussian_NPE_Node

# Define box and cosmology parameters
box_params = {'dim': 3, 'grid_res': 64, 'box_size': 1000.0}
cosmo_params = {'h': 0.67, 'Omega_cdm': 0.27, 'Omega_b': 0.05, ...}

# Initialize the inference node
node = Gaussian_NPE_Node(
    box_params=box_params,
    cosmo_params=cosmo_params,
    sigma_noise=0.1,
    num_epochs=30,
    batch_size=8,
    learning_rate=5e-3
)

# Train on data
await node.train(dataloader_train, dataloader_val)
```

### Inference

```python
# Load trained model
node.posterior.load_state_dict(torch.load('best_model.pt'))

# Get MAP estimate
z_MAP = node.get_z_MAP(x_observed)

# Sample from posterior using OU process
samples = node.posterior.sample(num_samples=100, x_obs=x_observed, steps=10000)
```

## Acknowledgments

This work builds upon:
- [FALCON framework](https://github.com/cweniger/falcon-zero) for simulation-based inference training
- [map2map](https://github.com/eelregit/map2map) for U-Net implementation
- [Pylians](https://github.com/franciscovillaescusa/Pylians3) for power spectrum analysis