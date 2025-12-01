import numpy as np
import asyncio
from time import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Import scheduler

from sbi.utils import BoxUniform
from sbi.neural_nets.net_builders import build_nsf

from .hypercubemappingprior import HypercubeMappingPrior
from .norms import LazyOnlineNorm

import torch.nn as nn
import torch.nn.functional as F


class LearnableNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))  # Use log(σ) for numerical stability

    def forward(self, x):
        std = torch.exp(self.log_std)
        return (x - self.mean) / std

    def inverse(self, z):
        std = torch.exp(self.log_std)
        return z * std + self.mean

    def log_abs_det_jacobian(self):
        return -self.log_std.sum()

    def get_std_normality_loss(self, z):
        # Regularize to encourage standard normal distribution
        mean_penalty = z.mean(0) ** 2
        std_penalty = (z.std(0, unbiased=False) - 1) ** 2
        return (mean_penalty + std_penalty).mean()


class Network(nn.Module):
    def __init__(self, theta, s, std_norm_reg=10.0):
        super().__init__()
        self.std_norm_reg = std_norm_reg
        self.norm = LearnableNorm(theta.shape[-1])
        self.net = build_nsf(theta.float(), s.float(), z_score_x=None, z_score_y=None)
        self.counter = 0

    def loss(self, theta, s):
        theta_norm = self.norm(theta)
        base_loss = self.net.loss(theta_norm.float(), condition=s.float())
        volume_log_det = self.norm.log_abs_det_jacobian()

        reg_loss = self.norm.get_std_normality_loss(theta_norm)
        total_loss = base_loss - volume_log_det + self.std_norm_reg * reg_loss

        self.counter += 1
        if self.counter % 20 == 0:
            #print(f"log|volume|: {volume_log_det.item():.4f}, reg_loss: {reg_loss.item():.4f}")
            print(f"Volume: {np.exp(volume_log_det.item()):.2e}, reg_loss: {reg_loss.item():.4f}")

        return total_loss

    def sample(self, num_samples, s):
        samples = self.net.sample((num_samples,), condition=s).detach()
        return self.norm.inverse(samples).detach()

    def log_prob(self, theta, s):
        theta_norm = self.norm(theta)
        log_prob = self.net.log_prob(theta_norm.float(), condition=s.float())
        volume_log_det = self.norm.log_abs_det_jacobian()
        return log_prob + volume_log_det



class NSFNode:
    def __init__(self, 
                 priors=[('uniform', -1., 1.)],
                 embeddings=None,
                 device='cpu',
                 num_epochs=10, 
                 lr_decay_factor=0.1,
                 scheduler_patience=4,
                 early_stop_patience=np.inf,
                 gamma = 0.5, 
                 lr=1e-2,
                 discard_samples=True,
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
        self.gamma = gamma
        self.lr = lr

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
        self._optimizer = AdamW(parameters, lr=self.lr)

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
                #print("Inference conditions:", len(inf_conditions))
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

    def conditioned_sample(self, num_samples, parent_conditions=[], evidence_conditions=[]):
        samples = self._aux_sample(num_samples, mode = 'posterior', parent_conditions = parent_conditions, evidence_conditions = evidence_conditions)
        return samples

    def proposal_sample(self, num_samples, parent_conditions=[], evidence_conditions=[]):
        samples = self._aux_sample(num_samples, mode = 'proposal', parent_conditions = parent_conditions, evidence_conditions = evidence_conditions)
        return samples

    def _aux_sample(self, num_samples, mode = None, parent_conditions=[], evidence_conditions=[]):
        """Sample from the proposal distribution given conditions."""
        inf_conditions = parent_conditions + evidence_conditions
        # Run conditions through summary network
        assert inf_conditions is not None, "Conditions must be provided."
        inf_conditions = [c.to(self.device) for c in inf_conditions]
        s = self._summary(inf_conditions, train=False)
        s, = self._align_singleton_batch_dims([s], length=num_samples)

        num_proposals = 256

        self._traindist.eval()
        samples_proposals = self._traindist.sample(num_proposals, s*0).detach()
        # (num_proposals, num_samples, theta_dim)

        self._posterior.eval()
        log_prob_post = self._posterior.log_prob(
            samples_proposals, s)  # (num_proposals, num_samples)

        self._traindist.eval()
        log_prob_dist = self._traindist.log_prob(
            samples_proposals, s*0)  # (num_proposals, num_samples)
        
        # Generate "mask" that equals one if samples are outside the [-1, 1] box
        mask = (samples_proposals < -2) | (samples_proposals > 2)
        mask = mask.any(dim=-1).float()*100     # (num_proposals, num_samples)

        gamma = self.gamma

        if mode == 'proposal':
            # Proposal samples, based on auxiliary distribution
            #log_weights = gamma/(1.+gamma)*log_prob_post - log_prob_dist - mask  # prior is uniform and hence neglected

            log_weights = gamma*(log_prob_post-log_prob_dist) - log_prob_dist - mask  # prior is uniform and hence neglected

        elif mode == 'posterior':
            # General posterior samples, based on auxiliary distribution alone
            log_weights = log_prob_post - 2*log_prob_dist - mask

            # General posterior samples, based on auxiliary distribution and reference probability
            #log_weights = log_prob_post - gamma/(1.+gamma)*log_prob_post_x0 - log_prob_dist - mask   # prior is uniform and hence neglected

            #log_weights = 1./(1.+gamma)*log_prob_post - log_prob_dist - mask   # prior is uniform and hence neglected (ONLY CORRECT FOR X = X0)

        elif mode == 'prior':
            # Prior samples, based on auxiliary distribution
            log_weights = -log_prob_dist - mask

        else:
            raise KeyError

        #  Use q(z) = q(z|x0)^gamma as proposal
        #  q(z|x) \propto p(x|z) q(z) is approximate posterior

        log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        weights = torch.exp(log_weights)  # (num_proposals, num_samples) - sum up to one in first dimension

        weights *= num_proposals
        #print("Weights:", weights.sum(dim=0))
        print("Effective samples:", ((weights**2).sum(dim=0)**0.5).min().item())

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
            #mask = log_ratio < self.log_ratio_threshold
            mask = log_ratio < np.inf
        else:
            mask = torch.zeros_like(log_ratio).bool()
        #print("rejection fraction:", mask.float().mean().item())
        return mask

        # p(z|x)/p_tilde(z) = p(x|z)/p_tilde(x) > 1e-3**dim_params
