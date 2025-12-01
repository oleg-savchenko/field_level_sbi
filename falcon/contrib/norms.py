import torch.nn as nn

class LazyOnlineNorm(nn.Module):
    def __init__(self, momentum=0.1, epsilon = 1e-20):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("running_mean", None)
        self.register_buffer("running_var", None)
        self.initialized = False
        self.epsilon = epsilon

    def forward(self, x):
        if not self.initialized:
            # Initialize running statistics based on the first minibatch
            self.running_mean = x.mean(dim=0).detach()
            self.running_var = x.var(dim=0, unbiased=False).detach() + self.epsilon**2
            self.initialized = True

        if self.training:
            # Compute batch mean and variance over batch dimension only
            batch_mean = x.mean(dim=0)  # Mean over batch dimension
            batch_var = x.var(dim=0, unbiased=False)  # Variance over batch dimension

            # Update running statistics (match shape explicitly)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

        #print("LazyOnlineNorm training flag:", self.training)

        # Ensure broadcasting by reshaping running stats before normalization
        return (x - self.running_mean) / (self.running_var.sqrt() + self.epsilon)

    def inverse(self, x):
        return x*self.running_var.sqrt() + self.running_mean

    def volume(self):
        return self.running_var.sqrt().prod()
