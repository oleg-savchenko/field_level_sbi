import numpy as np
import matplotlib.pyplot as plt
import torch
import falcon
from field_level_sbi import utils
from discodj import DiscoDJ
import ray

import os
import time
from datetime import datetime

print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")

########## BOX PARAMETERS ##########

box_params = {
        'box_size': 1000.,       #Mpc/h
        'grid_res': 64,         #resolution
        'h': 0.6711,
        'dim': 3
        }

# box = utils.Power_Spectrum_Sampler(box_parameters, device = 'cpu')

########## COSMO PARAMETERS ##########

cosmo_params = {
        'h': 0.6711,
        'Omega_b': 0.049,
        'Omega_cdm': 0.2685,
        # 'A_s': 2.1413e-09,
        'n_s': 0.9624,
        'non linear': 'halofit',
        'sigma8': 0.834,
    }

# Prior for the Gaussian NPE
# prior = box.get_prior_Q_factors(lambda k: torch.tensor(utils.get_pk_class(cosmo_params, 0, np.array(k)), device = 'cpu'))

########## SIMULATION PARAMETERS ##########

stepper = "bullfrog"
method = "pm"
res_pm = 2 * box_params['grid_res']
time_var = "D"
alpha = 1.5
theta = 0.5
antialias = 0
grad_kernel_order = 4
laplace_kernel_order = 0
n_resample = 1
numsteps = 1  # 10 steps
deconvolve = False
nlpt_order_ics = 2
worder = 2

a_ini = 1./(1+3.)
a_end = 1./(1+0.0)

sim_params = dict(a_ini=a_ini,
                  a_end=a_end,
                  stepper=stepper,
                  method=method,
                  res_pm=res_pm,
                  time_var=time_var,
                  alpha=alpha,
                  theta=theta,
                  antialias=antialias,
                  grad_kernel_order=grad_kernel_order,
                  laplace_kernel_order=laplace_kernel_order,
                  n_resample=n_resample,
                  deconvolve=deconvolve,
                  n_steps=numsteps,
                  nlpt_order_ics=nlpt_order_ics,
                  worder=worder)

########## SAMPLE ICs ##########

# class node_ics:
#     display_name = "Initial conditions"
#     def __init__(self, box_params=box_params):
#         import os
#         os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#         # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.1'
#         # os.environ["JAX_PLATFORMS"] = "cpu"
#         self.devices = self.jax.devices()
#         print(1, self.devices)
#         # self.device = "gpu" if np.any([d.platform == "gpu" for d in self.devices]) else "cpu"
#         self.device = 'cpu'
#         self.box_params = box_params
#         self.sample_ics = self.jax.jit(self._sample_ics_impl)
#         self.batched_sample_ics = self.jax.vmap(self.sample_ics)

#     @property
#     def jax(self):
#         import jax
#         return jax

#     @property
#     def jnp(self):
#         import jax.numpy as jnp
#         return jnp

#     def _sample_ics_impl(self, seed=1):
#         box_params = self.box_params
#         dim, res, boxsize = box_params['dim'], box_params['grid_res'], box_params['box_size']

#         dj = DiscoDJ(dim=3, res=res, boxsize=boxsize, device=self.device, cosmo='Quijote')
#         # print(dj)
#         dj = dj.with_timetables()
#         dj = dj.with_linear_ps()
#         dj = dj.with_ics(seed=seed)
#         return dj.get_delta_linear(a=1)

#     def sample(self, batch_size, parent_conditions=[]):
#         seeds = self.jnp.arange(batch_size)
#         deltas_ic = self.batched_sample_ics(seeds)
#         # deltas_ic = self.jax.lax.map(self.sample_ics, seeds, batch_size=32)
#         deltas_ic = torch.from_numpy(np.array(deltas_ic))
#         return deltas_ic

#     def get_shape_and_dtype(self):
#         return 3*(box_params['grid_res'],), 'float32'
    
########## FORWARD MODEL ##########

class node_forward:
    display_name = "Forward model"
    def __init__(self, box_params=box_params, sim_params=sim_params):
        # time.sleep(3)
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.85'
        self.devices = self.jax.devices()
        print(2, self.devices)
        self.device = self.devices[0] #"gpu" if np.any([d.platform == "gpu" for d in self.devices]) else "cpu"
        print(2, self.device)
        self.box_params = box_params
        self.sim_params = sim_params
        self.simulate = self.jax.jit(self._simulate_impl)
        self.batched_simulate = self.jax.vmap(self.simulate)

    @property
    def jax(self):
        import jax
        return jax

    @property
    def jnp(self):
        import jax.numpy as jnp
        return jnp

    def _simulate_impl(self, delta_ic):
        box_params = self.box_params
        dim, res, boxsize = box_params['dim'], box_params['grid_res'], box_params['box_size']

        dj = DiscoDJ(dim=3, res=res, boxsize=boxsize, device=self.device, cosmo='Quijote')
        # print(dj)
        dj = dj.with_timetables()
        dj = dj.with_external_ics(delta=delta_ic)
        dj = dj.with_lpt(n_order=nlpt_order_ics, grad_kernel_order=0)

        X_sim, _, _ = dj.run_nbody(**self.sim_params, use_diffrax=False)
        delta_fin = dj.get_delta_from_pos(X_sim, res=dj.res, worder=worder, antialias=1, deconvolve=True)
        return delta_fin

    def sample(self, batch_size, parent_conditions=[]):
        deltas_ic = parent_conditions[0]
        deltas_ic = self.jax.numpy.array(deltas_ic.cpu().numpy())
        # deltas_fin = self.batched_simulate(deltas_ic)
        deltas_fin = self.jax.lax.map(self.simulate, deltas_ic, batch_size=32)
        deltas_fin = torch.from_numpy(np.array(deltas_fin))
        return deltas_fin

    def get_shape_and_dtype(self):
        return 3*(box_params['grid_res'],), 'float32'
    
########## OBSERVATION ##########

class node_obs:
    display_name = "Noisy observed field"
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.cone_mask = torch.tensor(utils.create_cone_mask(fov_angle=[53.13], res=box_params['grid_res']))

    def sample(self, batch_size, parent_conditions=[]):
        deltas_fin = parent_conditions[0]
        deltas_obs = deltas_fin * self.cone_mask  # Apply cone mask
        deltas_obs = deltas_fin + torch.randn_like(deltas_fin) * self.sigma
        return deltas_obs
    
    def get_shape_and_dtype(self):
        return 3*(box_params['grid_res'],), 'float32'

################
# Main function
################

def main():
    n_train = 2048
    num_epochs = 50
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    filepath = '/home/osavchenko/falcon_oleg/zarr_stores/os250718_134112_disco_dj.zarr'
    run_name = 'Chebyshev_50ep_2ksims_lr_5e-3_Qx_masked_init_with_mask'
    sigma_noise = 0.1
    os.makedirs('./plots/'+current_time+'_'+run_name, exist_ok=True)
  
    Node_delta_ic = falcon.Node("delta_ic",
        falcon.LazyLoader("gaussian_npe.gaussian_npe_network_disco_dj_CG_Chebyshev_Qx_tests_masked.Gaussian_NPE_Node"),
        parents = [], evidence=["delta_fin",],
        module_config=dict(box_params = box_params, cosmo_params = cosmo_params, num_epochs = num_epochs, batch_size = 16, k_cut = 0.03, sigma_noise = sigma_noise, learning_rate = 5e-3, current_time = current_time, run_name = run_name),
        actor_config=dict(num_gpus=1)
        )
    
    Node_delta_fin = falcon.Node("delta_fin",
        node_forward,
        parents = ['delta_ic'],
        module_config=dict(box_params=box_params, sim_params=sim_params),)
        # actor_config=dict(num_gpus=.45)
        # )

    # Node_delta_obs = falcon.Node("delta_obs",
    #     node_obs,
    #     parents = ['delta_fin'],
    #     module_config = {'sigma': 0.1}
    #     )

    # Register graph nodes (ordering is irrelevant)
    graph = falcon.Graph([Node_delta_ic, Node_delta_fin])#, Node_delta_obs])

    print(graph)  # Show graph structure

    deployed_graph = falcon.DeployedGraph(graph)  # Deploy graph nodes as ray actors

    # samples = deployed_graph.sample(2)
    # print("\nSamples:")
    # for k, v in samples.items():
    #     # Print sample key name and shape
    #     print(k, ":", v.shape, type(v), v.device)
    
    # torch.save(samples, "samples.pt")
    # samples = torch.load("samples.pt")
    # observations = {
    #     "x": samples["delta_z0"][0].unsqueeze(0).to(device),
    # }
    # obs_shape = observations["x"].shape
    # print(f"observation_shape = {obs_shape}")
    # observations = {
    #     "delta_fin": torch.randn(1, 64, 64, 64),
    # }

    mask = torch.tensor(utils.create_cone_mask(fov_angle=[53.13], res=box_params['grid_res']), device='cuda')
    sample_obs = torch.load('sample_obs_disco_dj.pt')
    delta_fin = torch.from_numpy(sample_obs['delta_fin'].astype('f')).cuda()
    delta_fin = delta_fin * mask  + torch.randn_like(delta_fin) * sigma_noise  # Apply cone mask and noise
    observations = {
        "delta_fin": delta_fin,
    }
    sample_obs['delta_fin'] = delta_fin.cpu().numpy().astype('f')  # Convert to numpy for saving
    
    # 1) Prepare dataset manager for deployed graph and store initial samples
    shapes_and_dtypes = deployed_graph.get_shapes_and_dtypes()
    dataset_manager = falcon.get_zarr_dataset_manager(shapes_and_dtypes, filepath,
            num_min_sims = n_train, num_val_sims=256)
    #time.sleep(1)
    # ray.get(dataset_manager.generate_samples.remote(deployed_graph, num_sims = n_train))


    deployed_graph.train(dataset_manager, observations)

    # 4) Evaluation and storage (here sample from the trained graph)
    samples = deployed_graph.conditioned_sample(100, observations)['delta_ic']
    z_MAP = ray.get(deployed_graph.wrapped_nodes_dict["delta_ic"].call_method.remote('get_z_MAP', x_obs = delta_fin))
    torch.save(z_MAP, "./plots/"+current_time+"_"+run_name+"/z_MAP_"+current_time+"_"+run_name+".pt")

    utils.plot_samples_analysis(sample_obs, samples, z_MAP, box_params, cosmo_params, time=current_time, run_name=run_name)

    # 5) Clean up Ray resources
    deployed_graph.shutdown()

if __name__ == "__main__":
    main()
    ray.shutdown()  # Necessary to clean all the workers up
