import numpy as np
import matplotlib.pyplot as plt
import torch
import Pk_library as PKL
from classy import Class
import os

def get_pk_class(cosmo_params, z, k, non_lin = False):
    """While cosmo_params is the cosmological parameters, z is a single redshift,
    while k is an array of k values in units of h/Mpc.
    The returned power spectrum is in units of (Mpc/h)³.
    """
    h = cosmo_params['h']
    cosmo_params.update({
        'output': 'mPk',
        'P_k_max_h/Mpc': np.max(k),
        'z_max_pk': z,
    })
    cosmo = Class() 
    cosmo.set(cosmo_params)
    cosmo.compute()
    if non_lin:
        pk_class = h**3*np.array([cosmo.pk(h*ki, z) for ki in k])
    else:
        pk_class = h**3*np.array([cosmo.pk_lin(h*ki, z) for ki in k])
    return pk_class

def growth_D_approx(cosmo_params, z):
    Om0_m = cosmo_params['Omega_cdm'] + cosmo_params['Omega_b']
    Om0_L = 1. - Om0_m
    Om_m = Om0_m * (1.+z)**3 / (Om0_L + Om0_m * (1.+z)**3)
    Om_L = Om0_L/(Om0_L+Om0_m*(1.+z)**3)
    return ((1.+z)**(-1)) * (5. * Om_m/2.) / (Om_m**(4./7.) - Om_L + (1.+Om_m/2.)*(1.+Om_L/70.))

# def get_k(box_params, device='cuda'):
#     """Set up the 3D k-vector Fourier grid and calculate its magnitude for each point of the grid.
#     """
#     box_size = box_params['box_size']
#     N = box_params['grid_res']
#     d = box_size / (2*np.pi*N)
#     freq = torch.fft.fftfreq(N, d = d, device = device)
#     kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing = 'ij')
#     k = (kx**2 + ky**2 + kz**2)**0.5
#     k[0,0,0] = k[0,0,1]*1e-9    # Offset to avoid singularities (i.e., now k has no entries with zeros)
#     return k

def hartley(x, dim = (-3, -2, -1)):
    """
    Calculates the Hartley transform of the input field.
    axes: which dimensions to perform transformation on.
    """
    fx = torch.fft.fftn(x, dim = dim, norm = 'ortho')
    return (fx.real - fx.imag)

def whiten(x, prior):
    """
    Whiten a field with a given power spectrum defined by prior Q factors.
    """
    UT, D, U = prior[0], prior[1], prior[2]
    return UT(D**0.5 * U(x))

class Power_Spectrum_Sampler:
    def __init__(self, box_params, device = 'cuda', dim = 3):
        self.box_size = box_params['box_size']
        self.N = box_params['grid_res']
        self.shape = dim * (self.N,)
        self.hartley_dim = tuple(range(-dim, 0, 1))
        self.dim = dim
        self.device = device
        self.k = self.get_k(device = device)
        self.k_Nq = np.pi * self.N / self.box_size
        self.k_F = 2 * np.pi / self.box_size

    def get_k(self, device = None):
        """Set up the 3D k-vector Fourier grid and calculate its magnitude for each point of the grid.
        """
        if device is None:
            device = self.device
        d = self.box_size / (2*np.pi*self.N)
        freq = torch.fft.fftfreq(self.N, d = d, device = device)
        kx, ky, kz = torch.meshgrid(freq, freq, freq, indexing = 'ij')
        k = (kx**2 + ky**2 + kz**2)**0.5
        k[0,0,0] = k[0,0,1] #*1e-9    # Offset to avoid singularities (i.e., now k has no entries with zeros)
        return k

    def get_prior_Q_factors(self, pk):
        """Return components of the prior precision matrix.

        Q_prior = UT * D * U

        Returns:
            UT, D, U: Linear operator, tensor, linear operator.
        """
        D = (pk(self.k.cpu().flatten()) * (self.N/self.box_size)**self.dim)**-1
        U = lambda x: hartley(x, dim = self.hartley_dim).flatten(-len(self.shape), -1)
        UT = lambda x: hartley(x.unflatten(-1, self.shape), dim = self.hartley_dim)
        return UT, D, U

    def sample(self, num_samples, pk = None, prior = None):
        """Sample a Gaussian random field with a given power spectrum.
        """
        if prior is None:
            prior = self.get_prior_Q_factors(pk)
        UT, D, U = prior[0], prior[1], prior[2]
        if num_samples == 1:
            r = torch.randn(D.shape, device = self.device)
        else:
            r = torch.randn(num_samples, *D.shape, device = self.device)
        x = UT(r * D**-0.5)
        return x

    def top_hat_filter(self, x, k_min = None, k_max = None):
        """Sharp cutoff filter in Fourier space.
        """
        if k_max == None:
            mask = (self.k <= k_min)
        elif k_min == None:
            mask = (self.k >= k_max)
        else:
            mask = ((self.k <= k_min) & (self.k >= k_max))
        mask.to(self.device)
        return hartley(mask * hartley(x, dim = self.hartley_dim), dim = self.hartley_dim)
    
    def sigmoid_filter(self, x, k_cut, w_cut):
        """Sigmoidal high-pass filter in Fourier space centred at k_cut with width w_cut.
        """
        mask = torch.sigmoid((self.k - k_cut)/w_cut).to(x.device)
        # mask.to(self.device)
        return hartley(mask * hartley(x, dim = self.hartley_dim), dim = self.hartley_dim)
    
    def get_pk_pylians(self, delta, MAS = None):
        """
        Compute the power spectrum of an input field using the Pylians library.
        """
        Pk = PKL.Pk(delta, self.box_size, axis=0, MAS=MAS, threads=1, verbose=False)    # Compute power spectrum

        # Pk is a python class containing the 1D, 2D, and 3D power spectra
        k_pylians = Pk.k3D    # 3D P(k)
        pk_pylians = Pk.Pk[:, 0]    # Monopole
        return k_pylians, pk_pylians

def plot_samples_analysis(sample_obs, samples, z_MAP, box_params, cosmo_params, MAS=None, run_name='', time='', mask=1):
    """
    Plot the samples and their analysis.
    """
    os.makedirs("./plots/"+time+"_"+run_name, exist_ok=True)

    plt.rcParams['figure.facecolor'] = 'white'

    std = samples.std(axis=0)
    std = std.cpu().numpy().astype('f')
    # z_MAP = samples.mean(axis=0)
    z_MAP = z_MAP.cpu().numpy().astype('f')
    if mask is not 1:
        mask = mask.cpu().numpy().astype('f')
    
    samples = samples.cpu().numpy().astype('f')
    sample = samples[0]
    delta_ic = sample_obs['delta_ic'].astype('f')#.cpu().numpy()
    # delta_fin = sample_obs['delta_fin'].astype('f')#.cpu().numpy()
    delta_fin = sample_obs['delta_obs'].astype('f')#.cpu().numpy()
    residual = sample - delta_ic
    print('Sample absolute standard deviation from the truth: 10^3 * std(delta_ic - sample) =', residual.std())

    fig, axes = plt.subplots(3, 3, figsize = (16, 18), sharey=True)
    vmin, vmax = -3, 3
    for i in range(3):
        s=40+10*i
        figure = axes[0,i].imshow(delta_ic[s], origin='lower', cmap='seismic', vmin=vmin, vmax=vmax)#, interpolation = 'bilinear')#, interpolation='gaussian')
        axes[0,i].set_title(f'Initial density field slice {s}')
        axes[0,i].set_xlabel('x (voxels)',fontsize=14)
        if i == 0:
            axes[0,i].set_ylabel('y (voxels)',fontsize=14)
        if i == 2:
            cbar_ax1 = fig.add_axes([axes[0, i].get_position().x1 + 0.01, axes[0, i].get_position().y0, 0.01, axes[0, i].get_position().height])
            plt.colorbar(figure, ax=axes[0, :], cax=cbar_ax1)
        
        
        figure = axes[1,i].imshow(sample[s], origin='lower', cmap='seismic', vmin=vmin, vmax=vmax)#, interpolation = 'bilinear')#, interpolation='gaussian')
        axes[1,i].set_title(f'Sample from the posterior slice {s}')
        axes[1,i].set_xlabel('x (voxels)',fontsize=14)
        if i == 0:
            axes[1,i].set_ylabel('y (voxels)',fontsize=14)
        if i == 2:
            cbar_ax1 = fig.add_axes([axes[1, i].get_position().x1 + 0.01, axes[1, i].get_position().y0, 0.01, axes[1, i].get_position().height])
            plt.colorbar(figure, ax=axes[1, :], cax=cbar_ax1)


        figure = axes[2,i].imshow(residual[s], origin='lower', cmap='seismic', vmin=vmin/2, vmax=vmax/2)#, interpolation='gaussian')
        axes[2,i].set_title(f'Residual slice {s}')
        axes[2,i].set_xlabel('x (voxels)',fontsize=14)
        if i == 0:
            axes[2,i].set_ylabel('y (voxels)',fontsize=14)
        if i == 2:
            cbar_ax1 = fig.add_axes([axes[2, i].get_position().x1 + 0.01, axes[2, i].get_position().y0, 0.01, axes[2, i].get_position().height])
            plt.colorbar(figure, ax=axes[2, :], cax=cbar_ax1)
    fig.savefig("./plots/"+time+"_"+run_name+"/samples_"+time+"_"+run_name+".png")


    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    slice_idx = 63  # You can adjust this index
    vmin, vmax = -3, 3

    # Plot 1: True Field (delta_ic)
    im0 = axes[0].imshow(delta_ic[slice_idx], origin='lower', cmap='seismic', vmin=vmin, vmax=vmax)
    axes[0].set_title(f'True Field Slice {slice_idx}')
    axes[0].set_xlabel('x (voxels)', fontsize=14)
    axes[0].set_ylabel('y (voxels)', fontsize=14)
    cbar_ax0 = fig.add_axes([axes[0].get_position().x1 + 0.01, axes[0].get_position().y0, 0.01, axes[0].get_position().height])
    plt.colorbar(im0, cax=cbar_ax0)

    # Plot 2: MAP Field (z_MAP)
    im1 = axes[1].imshow(z_MAP[slice_idx], origin='lower', cmap='seismic', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'MAP Estimate Slice {slice_idx}')
    axes[1].set_xlabel('x (voxels)', fontsize=14)
    cbar_ax1 = fig.add_axes([axes[1].get_position().x1 + 0.01, axes[1].get_position().y0, 0.01, axes[1].get_position().height])
    plt.colorbar(im1, cax=cbar_ax1)

    # Plot 3: Standard Deviation (std)
    # std_vmax = std[slice_idx].max()  # or set a fixed value like 1.0
    im2 = axes[2].imshow(std[slice_idx], origin='lower', cmap='Purples')#, vmin=0, vmax=std_vmax)
    axes[2].set_title(f'Posterior Std Slice {slice_idx}')
    axes[2].set_xlabel('x (voxels)', fontsize=14)
    cbar_ax2 = fig.add_axes([axes[2].get_position().x1 + 0.01, axes[2].get_position().y0, 0.01, axes[2].get_position().height])
    plt.colorbar(im2, cax=cbar_ax2)

    # plt.tight_layout()
    fig.savefig("./plots/"+time+"_"+run_name+"/std_"+time+"_"+run_name+".png")
    plt.show()

    box = Power_Spectrum_Sampler(box_params, device = 'cpu')

    # Calculating power spectra, transfer functions and cross-power spectra for the true field, MAP estimation and linear prediction
    k_min, k_max = 1e-4, 10  # h/Mpc
    k_lin = np.logspace(np.log10(k_min), np.log10(k_max), 100)
    pk_class_z0 = get_pk_class(cosmo_params, 0, k_lin)

    k_pylians, pk_pylians_ic = box.get_pk_pylians(delta_ic, MAS=MAS)

    # MAP = z_MAP.cpu().numpy().astype('f')
    k_pylians, pk_pylians_MAP = box.get_pk_pylians(z_MAP, MAS=MAS)
    tk_pylians_MAP = np.sqrt(pk_pylians_MAP/pk_pylians_ic)

    Pk = PKL.XPk([z_MAP, delta_ic * mask], box_params['box_size'], axis=0, MAS=[MAS, MAS], threads=1)
    Pk0_1  = Pk.Pk[:,0,0]
    Pk0_2  = Pk.Pk[:,0,1]
    Pk0_X  = Pk.XPk[:,0,0]

    xpk_pylians_MAP = Pk0_X / (Pk0_1 * Pk0_2)**0.5


    Pk = PKL.XPk([delta_ic, delta_fin], box_params['box_size'], axis=0, MAS=[MAS, MAS], threads=1)
    Pk0_1  = Pk.Pk[:,0,0]
    Pk0_2  = Pk.Pk[:,0,1]
    Pk0_X  = Pk.XPk[:,0,0]

    xpk_pylians_linear = Pk0_X / (Pk0_1 * Pk0_2)**0.5


    # Calculating power spectra, transfer functions and cross-power spectra for samples
    pks = []
    tks = []
    xpks = []

    for i in range(len(samples)):
        delta_ic_samples = samples[i]#.astype('float32')

        k_pylians, pk_pylians_draws = box.get_pk_pylians(delta_ic_samples, MAS=MAS)
        Pk = PKL.XPk([delta_ic_samples, delta_ic], box_params['box_size'], axis=0, MAS=[MAS, MAS], threads=1)
        Pk0_1  = Pk.Pk[:, 0, 0]
        Pk0_2  = Pk.Pk[:, 0, 1]
        Pk0_X  = Pk.XPk[:, 0, 0]

        pks.append(pk_pylians_draws)
        tks.append(np.sqrt(pk_pylians_draws/pk_pylians_ic))
        xpks.append(Pk0_X / (Pk0_1 * Pk0_2)**0.5)

    pks = np.array(pks)
    tks = np.array(tks)
    xpks = np.array(xpks)

    k_Nq = box.k_Nq

    fig, axs = plt.subplots(3, sharex=True, sharey=False, height_ratios=[2, 1, 1])#, layout='compressed')
    # Plot power spectra of truth vs generated samples
    h = 4
    w = 4
    fig.set_size_inches((w, h*2))
    axs[0].plot(k_pylians, pk_pylians_ic, marker='.', markersize=0.5, label=r'True', linewidth=0.5, zorder=1e6)
    axs[0].plot(k_pylians, pks.mean(0), label=r'Inferred', linewidth=0.5, color='forestgreen')
    axs[0].fill_between(k_pylians, pks.mean(0) - pks.std(0), pks.mean(0) + pks.std(0), alpha=0.75, color='forestgreen')#color='#82A8D1')
    axs[0].fill_between(k_pylians, pks.mean(0) - 2*pks.std(0), pks.mean(0) + 2*pks.std(0), alpha=0.25, color='forestgreen')#color='#82A8D1')
    axs[0].plot(k_pylians, pk_pylians_MAP, color='magenta', label=r'MAP', alpha=0.75, linewidth=0.5)
    axs[0].plot(k_lin, pk_class_z0, label=r'Linear', color='black', alpha=0.3, linewidth=0.5)
    axs[0].axvline(x = k_Nq, color='r', linestyle='--', label=r'$k_{\rm{Nyq}}$', linewidth=0.5, alpha=0.5)
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    # axs[0].tick_params(axis='x', which='both', length=0)
    # axs[0].tick_params(axis='x', which='both', length=1)
    axs[0].set_ylabel(r"$P(k)$", fontsize=16)
    axs[0].legend(facecolor='white', edgecolor='none', framealpha=0.8)
    axs[0].set_ylim([5e2, 5e4])
    axs[0].set_xlim(left=k_pylians[0], right=k_Nq + 0.075)#1.1*k_pylians[-1])
    axs[0].grid(which='both', alpha=0.125)
    axs[0].yaxis.set_tick_params(labelright=True, labelleft=False)

    # Plot transfer function of sample
    axs[1].plot(k_pylians, tks.mean(0), color='forestgreen')#'#82A8D1')
    axs[1].fill_between(k_pylians, tks.mean(0) - tks.std(0), tks.mean(0) + tks.std(0), alpha=0.75, color='forestgreen')#'#82A8D1')
    axs[1].fill_between(k_pylians, tks.mean(0) - 2*tks.std(0), tks.mean(0) + 2*tks.std(0), alpha=0.25, color='forestgreen')#'#82A8D1')
    axs[1].plot(k_pylians, tk_pylians_MAP, color='magenta', alpha=0.75, linewidth=0.5)
    axs[1].axvline(x = k_Nq, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    axs[1].axhline(1.0, color='k', ls='--', lw=0.5)
    axs[1].set_xscale('log')
    axs[1].set_ylabel(r"$T(k)$", fontsize=16)
    # axs[1].tick_params(axis='x', which='both', length=1)
    axs[1].set_ylim(bottom=0.95, top=1.1)
    axs[1].set_ylim(0.93, 1.07)
    axs[1].set_xlim(left=k_pylians[0])
    axs[1].grid(which='both', alpha=0.1)
    # axs[1].legend(facecolor='white', edgecolor='none', framealpha=0.8)
    axs[1].set_yticks([0.95, 1, 1.05])
    axs[1].yaxis.set_tick_params(labelright=True, labelleft=False)

    # Plot cross-correlation of true vs samples
    axs[2].plot(k_pylians, xpk_pylians_MAP, color='magenta', alpha=0.75, linewidth=0.5)
    axs[2].plot(k_pylians, xpk_pylians_linear, alpha=0.75, linewidth=0.5, color='orange', label=r'$z=0$')
    axs[2].plot(k_pylians, xpks.mean(0), color='forestgreen', linewidth=0.25)#color='#82A8D1')
    axs[2].fill_between(k_pylians, xpks.mean(0) + xpks.std(0), xpks.mean(0) - xpks.std(0), alpha=0.75, color='forestgreen')#color='#82A8D1')
    axs[2].fill_between(k_pylians, xpks.mean(0) + 2*xpks.std(0), xpks.mean(0) - 2*xpks.std(0), alpha=0.25, color='forestgreen')#color='#82A8D1')
    axs[2].axvline(x = k_Nq, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
    axs[2].axhline(1.0, color='k', ls='--', lw=0.5)
    # axs[2].axhline(0, color='k', alpha=0.1, linewidth=0.75)#, linestyle='--')
    axs[2].set_xscale('log')
    # axs[2].tick_params(axis='x', which='both', length=1)
    # axs[2].tick_params(axis='x', which='both', length=0)
    axs[2].set_ylabel(r"$C(k)$", fontsize=16)
    axs[2].set_xlabel(r"$k$ [$h / \rm{Mpc}$]", fontsize=14)
    axs[2].set_ylim([-0.2, 1.2])
    axs[2].grid(which='both', alpha=0.1)
    axs[2].legend(facecolor='white', edgecolor='none', framealpha=0.9, loc="lower left")
    axs[2].set_yticks([0, 0.5, 1])
    axs[2].yaxis.set_tick_params(labelright=True, labelleft=False)

    plt.subplots_adjust(hspace=0)
    fig.savefig("./plots/"+time+"_"+run_name+"/sum_"+time+"_"+run_name+".png")#, bbox_inches='tight')

def plot_Q_matrix(K, network, time='', run_name=''):
    os.makedirs("./plots/"+time+"_"+run_name, exist_ok=True)
    plt.figure()
    plt.title(r'Q matrix diagonal values as a function of k, $Q_{like} = U^T D \, U$')
    plt.xlabel(r'$k~[h{\rm Mpc}^{-1}]$')
    plt.ylabel(r'$D(k)$')

    K = K.cpu().numpy().flatten()

    k_Nq = network.box.k_Nq
    # K = get_k().flatten()
    mask = (K < k_Nq) * (K > 1e-3)
    print(mask.sum())
    plt.scatter(K[mask][::100], network.Q_like.D.detach().flatten().cpu().numpy()[mask][::100], s=1, label=r'$D_{like}$')
    plt.scatter(K[mask][::100], network.Q_prior.D.detach().flatten().cpu().numpy()[mask][::100], s=1, label=r'$D_{prior}$')
    plt.scatter(K[mask][::1], network.Q_like.D.detach().flatten().cpu().numpy()[mask][::1] + network.Q_prior.D.detach().cpu().numpy()[mask][::1], s=1, label=r'$D_{posterior}$', alpha=0.5)
    plt.axvline(x = k_Nq, color='r', linestyle='--', label='$k_{Nyq}$')

    plt.legend(loc='best')
    plt.xscale('log')
    plt.savefig("./plots/"+time+"_"+run_name+"/Q_FFT_matrix_"+time+"_"+run_name+".png")
    plt.show()
    # plt.ylim([5e-3, k_Nq])
    # plt.yscale('log')
    plt.ylim([0, 100])

def plot_Q_R_matrix(D_values, time='', run_name=''):
    os.makedirs("./plots/"+time+"_"+run_name, exist_ok=True)
    # D_values = network.Q_like.D.detach().cpu().reshape(network.shape).numpy()

    # Plot masked data 

    plt.rcParams['figure.facecolor'] = 'white'
    fig, axes = plt.subplots(1, 2, figsize = (9, 5), sharey = True)

    s=32
    figure = axes[0].imshow(D_values[s], origin='lower', cmap='Purples')#, vmin=-3, vmax=3)
    axes[0].set_title(f'D values slice {s}')
    axes[0].set_xlabel('x (voxels)', fontsize=14)
    axes[0].set_ylabel('y (voxels)', fontsize=14)
    plt.colorbar(figure, ax=axes[0], fraction=0.046, pad=0.04)

    s=63
    figure = axes[1].imshow(D_values[s], origin='lower', cmap='Purples')
    axes[1].set_title(f'D values slice {s}')
    axes[1].set_xlabel('x (voxels)', fontsize=14)
    # axes[1].set_ylabel('y (voxels)', fontsize=14)
    plt.colorbar(figure, ax=axes[1], fraction=0.046, pad=0.04)

    fig.savefig("./plots/"+time+"_"+run_name+"/D_vals_"+time+"_"+run_name+".png", bbox_inches='tight')

def create_cone_mask(fov_angle=[53.13], res=64):
    """
    Create a mask with one/two cone(s) for a 3D NumPy array, defined by field-of-view angle(s).

    Parameters:
    fov_angle: list of full field-of-view angles (in degrees) for the cones (one or two values)
               if two cones: first value = upward-facing cone, second value = downward-facing cone
    res (int): size of the cube
    
    Returns:
    numpy.ndarray: A 3D mask with 1 inside the cone(s) and 0 outside
    """

    fov_angle = np.array(fov_angle)
    # Center of the cube
    center = (res - 1) / 2
    # Convert angle to radians and take half 
    theta = np.radians(fov_angle / 2)
    # Compute slope of the cone
    tan_theta = np.tan(theta)

    # Create z range
    z = np.arange(res) 
    # Create x, y coordinate grids (centered at 0)
    y, x = np.meshgrid(np.arange(res) - center, 
                       np.arange(res) - center, indexing='ij')
    # Compute radial distance
    r = np.sqrt(x**2 + y**2)

    if len(fov_angle)==1:  # One-cone mask
        # Create cone mask 
        # Cone equation: sqrt(x^2 + y^2) <= tan(theta) * z
        cone_mask = np.sqrt(x**2 + y**2) <= (tan_theta[0] * z[:, None, None])

    elif len(fov_angle)==2:  # Two-cones mask
        # Center z range at 0
        z = z - center

        # Create masks for both cones
        # Cone equation: sqrt(x^2 + y^2) <= tan(theta) * z
        mask_up = (z[:, None, None] > 0) & (r <= tan_theta[0] * z[:, None, None])  # Upward cone
        mask_down = (z[:, None, None] < 0) & (r <= tan_theta[1] * -z[:, None, None])  # Downward cone

        # Combine both cones into a single mask
        cone_mask = mask_up | mask_down

    return cone_mask.astype(int)
