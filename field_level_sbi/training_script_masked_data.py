import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from discodj import DiscoDJ
from map2map.models import UNet
from field_level_sbi import gaussian_npe_model, gaussian_npe_training, utils

import asyncio
from time import time
from datetime import datetime
import os

from typing import Callable, Tuple, Optional

class Gaussian_NPE_Network(torch.nn.Module):
    def __init__(self, box, prior, rescaling_factor=1, k_cut=0, w_cut=1e-3, batch_size=8):
        super().__init__()
        self.box = box
        self.N = box.N
        self.shape = 3*(self.N,)
        self.rescaling_factor = rescaling_factor
        self.k_cut = k_cut
        self.w_cut = w_cut
        self.batch_size = batch_size

        self.scale = torch.nn.Parameter(torch.ones(self.N**3))
        self.unet_mu = UNet(1, 1, hid_chan=8, bypass=False)
        self.unet_Q = UNet(1, 1, hid_chan=8, bypass=False)

        self.prior = prior
        self.Q_prior = gaussian_npe_model.Precision_Matrix_From_Factors(*prior)
        # self.Q_like = gaussian_npe.Precision_Matrix_FFT(self.N)
        # self.Q_like = gaussian_npe.Precision_Matrix_Real2(torch.ones(self.shape))
        self.Q_like = gaussian_npe_model.Precision_Matrix_Masked_FFT_mod(self.N)
        self.Q_post = lambda x: self.Q_prior(x) + self.Q_like(x)

    def estimator_mu(self, x):
        p3d = 6*(20,)
        xx = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        xx = self.unet_mu(xx.unsqueeze(1)).squeeze(1)
        
        x = x + self.box.sigmoid_filter(xx, self.k_cut, self.w_cut)
        x = self.prior[0](self.prior[2](x) * self.scale)    # the method also works fine without this factor, but it makes the results slightly better
        return x
    
    def estimator_Q(self, x):
        p3d = 6*(20,)
        x = F.pad(x.unsqueeze(0), p3d, "circular").squeeze(0)
        x = self.unet_Q(x.unsqueeze(1)).squeeze(1)
        return x

    def loss(self, x, z):
        """Calculates the loss function.
        """
        z = self.rescaling_factor**-1 * z
        z_MAP = self.estimator_mu(x)
        _D2 = self.estimator_Q(x)
        # print("D2", D2.shape)
        # self.Q_like = gaussian_npe.Precision_Matrix_Q1_Real(D_like)
        self.Q_like._D2 = _D2 #.unsqueeze(0)

        loss1 = 0.5 * ((z - z_MAP)*(self.Q_post(z - z_MAP))).mean() #.sum(dim=(-3, -2, -1)).mean()
        # loss2 = - 0.5 * torch.log(self.Q_like.D + self.Q_prior.D).mean(dim=-1)
        loss2 = -0.5 * gaussian_npe_training.stochastic_logdet_gradient(self.Q_post, self.shape, num_samples=self.batch_size)#, num_samples=z.shape[0])
        loss = loss1 + loss2 - loss2.detach()
        return loss

    def validation_loss(self, x, z):
        """Calculates the loss function with Chebyshev method.
        """
        with torch.no_grad():
            z = self.rescaling_factor**-1 * z
            z_MAP = self.estimator_mu(x)
            _D2 = self.estimator_Q(x)
            self.Q_like._D2 = _D2
            # print("D2", D2.shape)

            loss1 = 0.5 * ((z - z_MAP)*(self.Q_post(z - z_MAP))).mean().detach() # .sum(dim=(-3, -2, -1)).mean()
            # loss2 = - 0.5 * torch.log(self.Q_like.D + self.Q_prior.D).mean(dim=-1)
            shape = (len(z),) + self.shape
            # print("Vector shape", shape)
            loss2 = -0.5 * gaussian_npe_training.stochastic_logdet_chebyshev(self.Q_post, shape, num_samples=20, num_chebyshev=30) / self.N**3
            loss = loss1 + loss2
        return loss, loss1, loss2
    
    def get_z_MAP(self, x_obs):
        """Returns the MAP estimation for a given x_obs.
        """
        return self.estimator_mu(x_obs.unsqueeze(0))[0].squeeze(0).detach() #* self.rescaling_factor

    def sample(self, num_samples=1, x_obs = None, steps = 10_000, dt = 1e-4):#, gamma = 1):
        z_MAP = self.estimator_mu(x_obs.unsqueeze(0))
        z_MAP = z_MAP.squeeze(0).detach()
        _D2 = self.estimator_Q(x_obs.unsqueeze(0))
        _D2 = _D2.detach() #* self.rescaling_factor**-2
        # print(D_FFT.shape)
        self.Q_like._D2 = _D2

        # dt relates to the lambda_max(Q2) / dt ~ 1
        sqrt_dt = (2 * dt) ** 0.5

        # Math:
        # x_0 = N(0, Q2^{-1})
        # Iterate:
        #   noise ~ N(0, Q2^{-1})
        #   x_{t+dt} = x_t - dt * Q2^{-1} (Q1 + Q2) x_t + sqrt(2 dt) * noise

        # Un-preconditioned version:
        # noise ~ N(0, I)
        # x_{t+dt} = x_t - dt * (Q1 + Q2) x_t + sqrt(2 dt) * noise

        # x = self.Q_prior.sample(num_samples)
        x = self.box.sample(num_samples, prior = self.prior)
        x0 = x.clone()
        for _ in range(steps):
            noise = self.box.sample(num_samples, prior = self.prior) #self.Q_prior.sample(num_samples)
            x = x - dt * (self.Q_prior.apply_inverse(self.Q_like(x).detach()) + x) + sqrt_dt * noise
            # print(x[0, 0, 0, 0].item())
        x = z_MAP + x
        return x

class Gaussian_NPE_Node:
    display_name = "Initial density field"  
    def __init__(self, box_params, cosmo_params, sigma_noise=0.1,
                 rescaling_factor=1, k_cut=0, w_cut=1e-3,
                 num_epochs=60,
                 batch_size=8,
                 learning_rate=1e-2,
                 lr_decay_factor=0.1,
                 scheduler_patience=2,
                 early_stop_patience=5,
                 current_time='',
                 run_name=''):

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.85'
        self.jax_devices = self.jax.devices()
        print(2, self.jax_devices)
        self.jax_device = self.jax_devices[0] #"gpu" if np.any([d.platform == "gpu" for d in self.devices]) else "cpu"
        print(2, self.jax_device)
        self.device = 'cuda'

        self.box_params = box_params
        self.cosmo_params = cosmo_params
        self.box = utils.Power_Spectrum_Sampler(self.box_params, device = self.device)
        self.prior = self.box.get_prior_Q_factors(lambda k: torch.tensor(utils.get_pk_class(self.cosmo_params, 0, np.array(k)), device = self.device))
        self.sigma = sigma_noise # Noise level added to the final observed field on the fly during training
        self.mask = torch.tensor(utils.create_cone_mask(fov_angle=[53.13], res=box_params['grid_res']), device=self.device)

        self.sample_ics = self.jax.jit(self._sample_ics_impl)
        self.batched_sample_ics = self.jax.vmap(self.sample_ics)

        self.rescaling_factor = rescaling_factor
        self.k_cut = k_cut
        self.w_cut = w_cut
        self.posterior = Gaussian_NPE_Network(self.box, self.prior, self.rescaling_factor, self.k_cut, self.w_cut, batch_size).to(self.device)

        # Training hyperparameters
        self.num_epochs = num_epochs # Number of epochs
        self.batch_size = batch_size  # Batch size
        self.learning_rate = learning_rate # Starting learning rate
        self.lr_decay_factor = lr_decay_factor  # Factor to reduce LR
        self.scheduler_patience = scheduler_patience  # Patience for scheduler
        self.early_stop_patience = early_stop_patience  # Patience for early stopping
        
        # Initialize optimizer
        parameters = list(self.posterior.parameters())
        self._optimizer = AdamW(parameters, lr=self.learning_rate)

        # Initialize Learning Rate Scheduler
        self._scheduler = ReduceLROnPlateau(self._optimizer, mode='min', 
                                           factor=self.lr_decay_factor, 
                                           patience=self.scheduler_patience, 
                                           verbose=True)
        self.current_time = current_time
        self.run_name = run_name

    @property
    def jax(self):
        import jax
        return jax

    @property
    def jnp(self):
        import jax.numpy as jnp
        return jnp

    def _sample_ics_impl(self, seed=1):
        box_params = self.box_params
        dim, res, boxsize = box_params['dim'], box_params['grid_res'], box_params['box_size']

        dj = DiscoDJ(dim=3, res=res, boxsize=boxsize, device=self.jax_device, cosmo='Quijote')
        # print(dj)
        dj = dj.with_timetables()
        dj = dj.with_linear_ps()
        dj = dj.with_ics(seed=seed)
        return dj.get_delta_linear(a=1)

    def sample(self, batch_size, parent_conditions=[], seeds=None):
        """Sample from the prior distribution."""
        assert parent_conditions == [], "Conditions are not supported."
        if seeds == None:
            seeds = self.jnp.arange(batch_size)
        deltas_ic = self.batched_sample_ics(seeds)
        # deltas_ic = self.jax.lax.map(self.sample_ics, seeds, batch_size=32)
        deltas_ic = torch.from_numpy(np.array(deltas_ic))
        # print('a', type(deltas_ic))
        return deltas_ic

    def conditioned_sample(self, num_samples, parent_conditions=[], evidence_conditions=[]):
        """Sample from the posterior distribution given conditions."""
        assert parent_conditions == [], "Parent condition is not supported."
        x_obs = evidence_conditions[0].squeeze(0)
        return self.posterior.sample(num_samples, x_obs=x_obs)

    def get_shape_and_dtype(self):
        return 3*(self.box_params['grid_res'],), 'float32'

    def get_z_MAP(self, x_obs): #, parent_conditions=[], evidence_conditions=[]):
        # assert parent_conditions == [], "Parent condition is not supported."
        # x_obs = evidence_conditions[0].squeeze(0)
        x_obs = x_obs.to(self.device)
        return self.posterior.get_z_MAP(x_obs=x_obs)

    async def train(self, dataloader_train, dataloader_val, hook_fn=None):
        """Train the neural spline flow on the given data."""
        best_val_loss = float('inf')  # Best validation loss
        train_losses = []
        val_losses = []
        val_losses1 = []
        val_losses2_chebyshev = []
        lr_history = []

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            # Training loop
            loss_avg = 0
            num_samples = 0
            time_start = time()
            for batch in dataloader_train:
                self._optimizer.zero_grad()
                # if not self.networks_initialized:
                #     self._initialize_networks(u, inf_conditions)
                
                if num_samples == 0 and epoch == 0:
                    print("training is being done")
                    print(type(batch))
                    print(len(batch))
                    print(f"batch_size = {self.batch_size}")
                    for delta in batch:
                        print(delta.shape)
                
                _, z, x = batch[0], batch[1], batch[2]
                zc = z.to(self.device)
                xc = x.to(self.device)
                xc = x.to(self.device) * self.mask + torch.randn_like(x, device = self.device) * self.sigma  # Adding noise on the fly
                
                loss = self.posterior.loss(xc, zc)
                # loss = torch.mean(losses)
                loss.backward()
                self._optimizer.step()

                num_samples += len(batch)
                loss_avg += loss.sum().item()

                # Run hook and allow other tasks to run
                # if hook_fn is not None:
                #     hook_fn(self, batch)
                await asyncio.sleep(0)

            print("Duration training epoch:", time()-time_start)

            loss_avg /= num_samples
            print(f"Training loss: {loss_avg}")
            train_losses.append(loss_avg)

            # Validation loop
            val_loss_avg = 0
            val_samples = 0
            val_loss1_avg = 0
            val_loss2_chebyshev_avg = 0
            for batch in dataloader_val:
                _, z, x = batch[0], batch[1], batch[2]
                zc = z.to(self.device)
                xc = x.to(self.device)
                xc = x.to(self.device) * self.mask + torch.randn_like(x, device = self.device) * self.sigma  # Adding noise on the fly

                loss, loss1, loss2_chebyshev = self.posterior.validation_loss(xc, zc)

                val_samples += len(batch)
                val_loss_avg += loss.item()
                # val_loss_avg += loss2.item()  ### DELETE THIS LINE FOR OTHER TESTS
                val_loss1_avg += loss1.item()
                # val_loss2_avg += loss2.item()
                val_loss2_chebyshev_avg += loss2_chebyshev
                await asyncio.sleep(0)

            val_loss_avg /= val_samples
            val_loss1_avg /= val_samples
            # val_loss2_avg /= val_samples
            val_loss2_chebyshev_avg /= val_samples
            # val_loss2_stochastic_avg /= val_samples
            print(f"Validation loss: {val_loss_avg}")
            # print(f"Validation loss2: {val_loss2_avg}")
            # print(f"Validation loss2 Chebyshev: {val_loss2_chebyshev_avg}")
            val_losses.append(val_loss_avg)
            val_losses1.append(val_loss1_avg)
            # val_losses2.append(val_loss2_avg)
            val_losses2_chebyshev.append(val_loss2_chebyshev_avg)
            # val_losses2_stochastic.append(val_loss2_stochastic_avg)

            self._scheduler.step(val_loss_avg)
            lr_history.append(self._optimizer.param_groups[0]['lr'])
            print(f"Learning rate: {self._optimizer.param_groups[0]['lr']}")

            # Early Stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                epochs_no_improve = 0
                print("Validation loss improved.")
                best_model_params = self.posterior.state_dict()
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{self.early_stop_patience} epochs.")

            if epochs_no_improve >= self.early_stop_patience:
                print("Early stopping triggered.")
                break

            # # Save the best model parameters
            # torch.save(best_model_params, 'best_model.pt')
            # print("Best model saved with Validation Loss: {:.4f}".format(best_val_loss))
            # torch.save(self.posterior.Q_like.D.detach().reshape(self.posterior.shape).cpu().numpy(), 'D_values.pt')
            if (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                # Plot training and validation loss

                os.makedirs('./plots/'+self.current_time+'_'+self.run_name, exist_ok=True)
                plt.figure()
                plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss (1)')
                # plt.plot(range(1, len(train_losses1) + 1), train_losses1, label='Training Loss 1')
                # plt.plot(range(1, len(train_losses2) + 1), train_losses2, label='Training Loss 2')
                # plt.plot(range(1, len(train_losses2_chebyshev) + 1), train_losses2_chebyshev, label='Training Loss 2 Chebyshev', linestyle=':')
                # plt.plot(range(1, len(train_losses2_stochastic) + 1), train_losses2_stochastic, label='Training Loss 2 Stochastic', linestyle='-.')
                plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss (with Chebyshev)', linestyle='--')
                plt.plot(range(1, len(val_losses1) + 1), val_losses1, label='Validation Loss 1', linestyle='--')
                # plt.plot(range(1, len(val_losses2) + 1), val_losses2, label='Validation Loss 2', linestyle='--')
                plt.plot(range(1, len(val_losses2_chebyshev) + 1), val_losses2_chebyshev, label='Validation Loss 2 Chebyshev', linestyle=':')
                # plt.plot(range(1, len(val_losses2_stochastic) + 1), val_losses2_stochastic, label='Validation Loss 2 Stochastic', linestyle='-.')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss over Epochs')
                plt.legend()
                plt.grid(True)
                plt.savefig('./plots/'+self.current_time+'_'+self.run_name+'/loss'+'_'+self.current_time+'_'+self.run_name+'.png')
                plt.show()

                # --- Plot LR over epochs ---
                plt.figure()
                num_epochs = len(lr_history)
                plt.plot(range(1, num_epochs+1), lr_history, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.title('LR Schedule')
                plt.grid(True)
                plt.savefig('./plots/'+self.current_time+'_'+self.run_name+'/lr_schedule'+'_'+self.current_time+'_'+self.run_name+'.png')
                plt.show()

                torch.save(best_model_params, './plots/'+self.current_time+'_'+self.run_name+'/best_model_'+self.current_time+'_'+self.run_name+'.pt')

        self.posterior.eval()

        # D2_values = self.posterior.Q_like.outer_D2.detach().reshape(-1, *self.posterior.shape).cpu()
        # torch.save(D_values, './plots/'+self.current_time+'_'+self.run_name+'/D_values_'+self.current_time+'_'+self.run_name+'.pt')
        # utils.plot_Q_R_matrix(D2_values[0].numpy(), time=self.current_time, run_name=self.run_name)
        torch.save(best_model_params, './plots/'+self.current_time+'_'+self.run_name+'/best_model_'+self.current_time+'_'+self.run_name+'.pt')
        # utils.plot_Q_matrix(self.box.get_k(), self.posterior, time=self.current_time, run_name=self.run_name)
