import numpy as np
import asyncio
from time import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import scheduler

from sbi.utils import BoxUniform
from sbi.neural_nets.net_builders import build_nsf

from .hypercubemappingprior import HypercubeMappingPrior

#class FocusTrans:
#    def forward(self, x):
#        # Generate batch online normalised versions of x
#        # s are the normalising factors
#        # (Model learns to predict z, given s and context)
#        return z, s
#
#    def get_s(self):
#        # Returns reference normalisation s
#        return s
#
#    def backward(self, z):
#        # Inverse of forward, assuming reference s was used to generate z
#        return x

"""
z, scales = ft(theta)
s = s + lin(scales)
net.loss(z, conditions=s)

scales = ft.ref()
s = s + lin(scales)
z = net.sample(conditions=s)
theta = ft.inverse(z)
"""

class Network(torch.nn.Module):
    def __init__(self, theta, s):
        super(Network, self).__init__()
        self.net = build_nsf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
                                    #hidden_features=128, num_transforms=5,
                                    #hidden_layers_spline_context=3,
                                    #use_batch_norm=True,
                                    #num_block=3)

    def loss(self, theta, s):
        return self.net.loss(theta.float(), condition=s.float())

    def sample(self, num_samples, s):
        # Return (num_samples, num_conditions, theta_dim) - standard pyro 
        samples = self.net.sample((num_samples,), condition=s).detach()
        return samples

# Pyro convention: sample_shape + batch_shape + event_shape

    def log_prob(self, thetas, s):
        # Has to work for general (*batch_shape, theta_dim)
        # (num_proposals, num_conditions, theta_dim)
        log_prob = self.net.log_prob(thetas.float(), condition=s.float())  # (num_proposals, num_samples)
        return log_prob


class NSFNode:
    def __init__(self, 
                 priors=[('uniform', -1., 1.)],
                 embeddings=None,
                 device='cpu',
                 num_epochs=10, 
                 lr_decay_factor=0.1,
                 scheduler_patience=2,
                 early_stop_patience=6,
                 discard_samples=True
                 ):
        # Configuration
        self.param_dim = len(priors)
        self.embeddings = embeddings
        self.device = device
        self.num_epochs = num_epochs
        self.lr_decay_factor = lr_decay_factor  # Factor to reduce LR
        self.scheduler_patience = scheduler_patience  # Patience for scheduler
        self.early_stop_patience = early_stop_patience  # Patience for early stopping
        self.discard_samples = discard_samples

        # Prior distribution
        self._prior = HypercubeMappingPrior(priors)

        # Runtime variables
        self.log_ratio_threshold = -np.inf  # Dynamic threshold for rejection sampling
        self.networks_initialized = False

    def _initialize_networks(self, theta, conditions):
        inf_conditions = conditions
        print("Initializing LearnableDistribution...")
        print("GPU available:", torch.cuda.is_available())

        # Initialize embedding networks
        if self.embeddings is not None:
            self._embeddings = [
                (e().to(self.device) if e is not None else lambda x: x) for e in self.embeddings]

        # Initialize neural spline flow for posterior distribution
        inf_conditions = [c.to(self.device) for c in inf_conditions]
        s = self._summary(inf_conditions, train=False)
        theta = theta.to(self.device)
        self._posterior = Network(theta, s)
        self._posterior.to(self.device)

        # Initialize neural spline flow for training distribution
        self._traindist = Network(theta, s*0)
        self._traindist.to(self.device)

        # Initialize optimizer
        parameters = list(self._posterior.parameters())
        parameters += list(self._traindist.parameters())
        if self.embeddings is not None:
            for e in self._embeddings:
                # Can be optimized?
                # If e is object with parameters, add them to the list
                if hasattr(e, "parameters"):
                    parameters += list(e.parameters())
        self._optimizer = AdamW(parameters, lr=1e-2)

        # Initialize Learning Rate Scheduler
        self._scheduler = ReduceLROnPlateau(self._optimizer, mode='min', 
                                           factor=self.lr_decay_factor, 
                                           patience=self.scheduler_patience, 
                                           verbose=True)

        # Set flag
        self.networks_initialized = True
        print("...done initializing LearnableDistribution.")

    def _align_singleton_batch_dims(self, tensors, length=None):
        """Broadcast singleton batch dimensions of tensors in a list to same length."""
        if length is None:
            length = max([len(t) for t in tensors])
        return [t.expand(length, *t.shape[1:]) for t in tensors]

    def _summary(self, inf_conditions, train = True):
        """Run conditions through embedding networks and concatenate them."""
        if self.embeddings is not None:
            for e in self._embeddings:
                if hasattr(e, 'train') and hasattr(e, 'eval'):
                    if train: e.train()
                    else: e.eval()
            inf_conditions = [e(c) for e, c in zip(self._embeddings, inf_conditions)]
        inf_conditions = self._align_singleton_batch_dims(inf_conditions)
        s = torch.cat(inf_conditions, dim=1)  # Concatenate all conditions into one tensor
        return s

    def sample(self, num_samples, parent_conditions=[]):
        """Sample from the prior distribution."""
        assert parent_conditions == [], "Conditions are not supported."
        samples = self._prior.sample(num_samples)
        return samples

    def get_shape_and_dtype(self):
        """Return shape and dtype of the samples."""
        return (self.param_dim,), 'float64'

    async def train(self, dataloader_train, dataloader_val, hook_fn=None):
        """Train the neural spline flow on the given data."""
        best_val_loss = float('inf')  # Best validation loss
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            # Training loop
            loss_aux_avg = 0
            loss_train_avg = 0
            num_samples = 0
            time_start = time()
            for batch in dataloader_train:
                #_, theta, parent_conditions, evidence_conditions = batch[0], batch[1], batch[2]
                #inf_conditions = parent_conditions + evidence_conditions
                _, theta, inf_conditions = batch[0], batch[1], batch[2:]
                u = self._prior.inverse(theta)
                if not self.networks_initialized:
                    self._initialize_networks(u, inf_conditions)
                self._optimizer.zero_grad()
                inf_conditions = [c.to(self.device) for c in inf_conditions]
                s = self._summary(inf_conditions, train=True)
                uc = u.to(self.device)
                sc = s.to(self.device)

                self._posterior.train()
                losses = self._posterior.loss(uc, sc)
                loss_train = torch.mean(losses)

                self._traindist.train()
                losses = self._traindist.loss(uc, sc.detach()*0)
                loss_aux = torch.mean(losses)

                loss_total = loss_train + loss_aux
                loss_total.backward()
                self._optimizer.step()

                num_samples += len(batch)
                loss_train_avg += loss_train.sum().item()
                loss_aux_avg += loss_aux.sum().item()

                # Run hook and allow other tasks to run
                if hook_fn is not None:
                    hook_fn(self, batch)
                await asyncio.sleep(0)

            print("Duration training epoch:", time()-time_start)

            loss_train_avg /= num_samples
            loss_aux_avg /= num_samples
            print(f"Training loss  : {loss_train_avg}")
            print(f"Aux loss       : {loss_aux_avg}")

            # Validation loop
            val_loss_avg = 0
            val_samples = 0
            for batch in dataloader_val:
                #_, theta, parent_conditions, evidence_conditions = batch[0], batch[1], batch[2]
                #inf_conditions = parent_conditions + evidence_conditions
                _, theta, inf_conditions = batch[0], batch[1], batch[2:]
                u = self._prior.inverse(theta)
                inf_conditions = [c.to(self.device) for c in inf_conditions]
                s = self._summary(inf_conditions, train=False)
                uc = u.to(self.device)
                sc = s.to(self.device)

                self._posterior.eval()
                losses = self._posterior.loss(uc, sc)
                loss = torch.mean(losses)

                val_samples += len(batch)
                val_loss_avg += loss.sum().item()
                await asyncio.sleep(0)

            val_loss_avg /= val_samples
            print(f"Validation loss: {val_loss_avg}")

            self._scheduler.step(val_loss_avg)

            # Early Stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                epochs_no_improve = 0
                print("Validation loss improved.")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve}/{self.early_stop_patience} epochs.")

            if epochs_no_improve >= self.early_stop_patience:
                print("Early stopping triggered.")
                break

    def conditioned_sample(self, num_samples,
                           parent_conditions=[], evidence_conditions=[]):
        """Sample from the posterior distribution given conditions."""
        inf_conditions = parent_conditions + evidence_conditions
        # Run conditions through summary network
        assert inf_conditions is not None, "Conditions must be provided."
        inf_conditions = [c.to(self.device) for c in inf_conditions]
        s = self._summary(inf_conditions, train=False)
        assert len(s) == 1, "Only one condition supported so far."

        #num_conditions = len(s)

        #If num_samples > len(conditions), conditions are repeated.
        #num_samples = num_conditions if num_samples is None else num_samples
        #assert num_samples % num_conditions == 0, "Number of samples must be divisible by number of conditions."

        self._posterior.eval()
        samples = self._posterior.sample(num_samples, s).detach()
        # (num_samples, num_conditions, theta_dim)
        #samples = samples.flatten(start_dim=0, end_dim=1).to('cpu')
        samples = samples.squeeze(1)  # (num_samples, theta_dim)
        samples = samples.to('cpu')

        samples = self._prior.forward(samples)

        return samples

    def proposal_sample(self, num_samples, parent_conditions=[], evidence_conditions=[]):
        """Sample from the proposal distribution given conditions."""
        inf_conditions = parent_conditions + evidence_conditions
        # Run conditions through summary network
        assert inf_conditions is not None, "Conditions must be provided."
        inf_conditions = [c.to(self.device) for c in inf_conditions]
        s = self._summary(inf_conditions, train=False)
        s, = self._align_singleton_batch_dims([s], length=num_samples)

        num_proposals = 128

        # (num_proposals, num_samples, theta_dim)
        self._traindist.eval()
        samples_proposals = self._traindist.sample(num_proposals, s*0).detach()
        # (num_proposals, num_conditions, theta_dim)

        self._posterior.eval()
        log_prob_post = self._posterior.log_prob(
            samples_proposals, s)  # (num_proposals, num_samples)

        self._traindist.eval()
        log_prob_dist = self._traindist.log_prob(
            samples_proposals, s*0)  # (num_proposals, num_samples)
        
        log_ratio = log_prob_post - 0*log_prob_dist   # (num_proposals, num_samples)

        # Generate "mask" that equals one if samples are outside the [-1, 1] box
        mask1 = (samples_proposals < -1) | (samples_proposals > 1)
        mask1 = mask1.any(dim=-1).float()     # (num_proposals, num_samples)
        mask2 = (log_ratio<self.log_ratio_threshold).float()

        log_weights = -log_prob_dist - mask1*50 - mask2*50

        log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        weights = torch.exp(log_weights)  # (num_proposals, num_samples) - sum up to one in first dimension

        #weights *= num_proposals
        #print("Weights:", weights.sum(dim=0))
        #print("Effective samples:", (weights**2).sum(dim=0)**0.5)

        idx = torch.multinomial(weights.T, 1, replacement=True).squeeze(-1)

        # samples_proposals have shape (num_proposals, num_samples, theta_dim)
        # samples will have shape (num_samples, theta_dim)
        # idx has shape (num_samples,) and ranges from 0 to num_proposals-1
        # samples by samples_proposals[idx[i], i, :] for i in range(num_samples)

        samples = samples_proposals[idx, torch.arange(num_samples), :]

        samples = self._prior.forward(samples)

        return samples.to('cpu')

    def discardable(self, theta, parent_conditions=[], evidence_conditions=[]):
        inf_conditions = parent_conditions + evidence_conditions
        u = self._prior.inverse(theta)
        #print([c.shape for c in inf_conditions])
        #return torch.zeros(len(theta), dtype=torch.bool)
        inf_conditions = [c.to(self.device) for c in inf_conditions]
        s = self._summary(inf_conditions, train=False)
        u, s = self._align_singleton_batch_dims([u, s])
        u = u.to(self.device)
        self._posterior.eval()
        self._traindist.eval()
        log_prob1 = self._posterior.log_prob(u.unsqueeze(0), s).squeeze(0).to('cpu')
        log_prob2 = self._traindist.log_prob(u.unsqueeze(0), s*0).squeeze(0).to('cpu')
        log_ratio = log_prob1 - 0*log_prob2  #  p(z|x)/p(z)

        alpha = 0.99
        eta = 1e-3
        t = self.log_ratio_threshold
        t += eta*(sum((log_ratio > t)*(log_ratio - t)*alpha) - 
                  sum((log_ratio < t)*(t - log_ratio)*(1-alpha))
                  )
        offset = 0.5*3**2*self.param_dim
        self.log_ratio_threshold = max(log_ratio.max().item()-offset, self.log_ratio_threshold)
        
        if self.discard_samples:
            mask = log_ratio < self.log_ratio_threshold
        else:
            mask = torch.zeros_like(log_ratio).bool()
        #print("rejection fraction:", mask.float().mean().item())
        return mask

        # p(z|x)/p_tilde(z) = p(x|z)/p_tilde(x) > 1e-3**dim_params
