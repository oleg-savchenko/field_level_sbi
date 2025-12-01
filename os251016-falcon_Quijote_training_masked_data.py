import numpy as np
import matplotlib.pyplot as plt
import torch
import falcon
from field_level_sbi import utils
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

def growth_D_approx(z, params):
    Om0_m = params['Omega_cdm'] + params['Omega_b']
    Om0_L = 1. - Om0_m
    Om_m = Om0_m * (1.+z)**3 / (Om0_L + Om0_m * (1.+z)**3)
    Om_L = Om0_L/(Om0_L+Om0_m*(1.+z)**3)
    return ((1.+z)**(-1)) * (5. * Om_m/2.) / (Om_m**(4./7.) - Om_L + (1.+Om_m/2.)*(1.+Om_L/70.))

z_Quijote = 127
Dz127_approx = growth_D_approx(z_Quijote, cosmo_params)/growth_D_approx(0, cosmo_params)
print('Dz127_approx =', Dz127_approx)

# Prior for the Gaussian NPE
# prior = box.get_prior_Q_factors(lambda k: torch.tensor(utils.get_pk_class(cosmo_params, 0, np.array(k)), device = 'cpu'))

def path_to_file_z0(i):
    return f'/gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_fiducial_64/{i}/df_m_64_PCS_z=0.npy'

def path_to_file_z127(i):
    return f'/gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_fiducial_64/{i}/df_m_64_PCS_z=127.npy'

    
########## FORWARD MODEL ##########

class node_forward:
    display_name = "Forward model"
    def __init__(self, box_params=box_params, start_seed=0):
        self.seed = start_seed
        self.box_params = box_params
 
    def sample(self, batch_size, parent_conditions=[]):
        print("Seed_z0 = ", self.seed)
        deltas_ic = torch.from_numpy(np.load(path_to_file_z0(self.seed))).unsqueeze(0)
        self.seed += 1
        return deltas_ic

    def get_shape_and_dtype(self):
        return 3*(self.box_params['grid_res'],), 'float32'
    
################
# Main function
################

def main():
    n_train = 2048
    num_epochs = 30
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    filepath = '/gpfs/scratch1/shared/osavchenko/zarr_stores/Quijote_fiducial_64.zarr'
    run_name = 'Quijote_Chebyshev_30ep_2ksims_lr_5e-3_Qx_2unets'
    os.makedirs('./plots/'+current_time+'_'+run_name, exist_ok=True)
    sigma_noise = 0.1
  
    Node_delta_ic = falcon.Node("delta_ic",
        falcon.LazyLoader("gaussian_npe.gaussian_npe_network_disco_dj_CG_Chebyshev_Qx_tests2_2unets.Gaussian_NPE_Node"),
        parents = [], evidence=["delta_fin",],
        module_config=dict(box_params = box_params, cosmo_params = cosmo_params, num_epochs = num_epochs, batch_size = 16, k_cut = 0.03, sigma_noise = sigma_noise, rescaling_factor = Dz127_approx, learning_rate=5e-3, current_time = current_time, run_name = run_name),
        actor_config=dict(num_gpus=1)
        )
    
    Node_delta_fin = falcon.Node("delta_fin",
        node_forward,
        parents = ['delta_ic'],
        module_config=dict(box_params=box_params),)
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


    delta_fin = torch.from_numpy(np.load(path_to_file_z0(0)).astype('f')).cuda()
    delta_fin += torch.randn_like(delta_fin) * sigma_noise
    delta_ic = torch.from_numpy(np.load(path_to_file_z127(0)).astype('f')).cuda() / Dz127_approx

    sample_obs = {
        'delta_obs': delta_fin.cpu().numpy(),
        "delta_fin": delta_fin.cpu().numpy(),
        'delta_ic': delta_ic.cpu().numpy(),
    }
    observations = {
        "delta_fin": delta_fin,
    }
    
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
    # torch.save(z_MAP, "./plots/"+current_time+"_"+run_name+"/z_MAP_"+current_time+"_"+run_name+".pt")

    utils.plot_samples_analysis(sample_obs, samples, z_MAP, box_params, cosmo_params, MAS='PCS', time=current_time, run_name=run_name)

    # 5) Clean up Ray resources
    deployed_graph.shutdown()

if __name__ == "__main__":
    main()
    ray.shutdown()  # Necessary to clean all the workers up
