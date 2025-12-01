import torch
import torch.nn.functional as F
from map2map.models import UNet
from abc import ABC, abstractmethod
from .utils import hartley

class GDG_Factor_Matrix(torch.nn.Module, ABC):
    """Parametrization of a symmetric positive definite matrix.
    
    Q = G_T * D * G
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self):
        super().__init__()     
    
    def forward(self, x):
        """(B, N, N, N) --> (B, N, N, N)"""
        x = self.G(x)
        x = self.D*x
        x = self.G_T(x)
        return x
    
    def get_factors(self):
        G_T = self.G_T
        G = self.G
        D = self.D
        return G_T, D, G

    @abstractmethod
    def G(self, x):
        pass
    
    @property
    @abstractmethod
    def D(self):
        pass

    @abstractmethod
    def G_T(self, x):
        pass

class Precision_Matrix_From_Factors(GDG_Factor_Matrix):
    """Parametrization of a symmetric positive definite matrix.
    
    Q = G_T * D * G
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, G_T, D, G, requires_grad = False):
        super().__init__()
        self._G = G
        self._G_T = G_T
        self._D = torch.nn.Parameter(D, requires_grad = requires_grad)

    def G(self, x):
        return self._G(x)
    
    @property
    def D(self):
        return self._D
    
    def G_T(self, x):
        return self._G_T(x)

    def apply_inverse(self, x):
        """(B, N, N, N) --> (B, N, N, N)"""
        x = self.G(x)
        x = self.D**-1 * x
        x = self.G_T(x)
        return x

    def sample(self, num_samples=1, whiten=False):
        z = torch.randn(num_samples, *self.D.shape, device=self.D.device)
        # z = self.G(z)
        if not whiten:
            z = z * self.D.detach()**-0.5  # apply sqrt covariance in frequency domain
            z = self.G_T(z)
        return z

class Precision_Matrix_FFT(GDG_Factor_Matrix):
    """Parametrization of a symmetric positive definite matrix diagonal in Fourier space.
    
    Q = F_T * D * F
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, N, device='cuda'):
        super().__init__()
        self.N = N
        self.shape = 3 * (self.N,)
        self.device = device
        # self.logD = torch.nn.Parameter(torch.ones(self.N**3))    # tried in the past, sqrt works better usually
        self.sqrt_D = torch.nn.Parameter(torch.ones(self.N**3, device=device))

    def G(self, x):
        x = hartley(x).flatten(-len(self.shape), -1)
        return x
    
    @property
    def D(self):
        # return torch.exp(self.logD)
        return self.sqrt_D**2

    def G_T(self, x):
        x = hartley(x.unflatten(-1, self.shape))
        return x

class Precision_Matrix_Masked_FFT(GDG_Factor_Matrix):
    """Parametrization of a symmetric positive definite matrix.
    
    Q = D2 * U_H * D * U_H * D2
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, N, device='cuda'):
        super().__init__()
        self.N = N
        self.shape = 3*(self.N,)
        # self.logD = torch.nn.Parameter(torch.ones(self.N**3))    # tried in the past, sqrt works better usually
        # self.logD2 = torch.nn.Parameter(torch.ones(self.shape))
        self.device = device
        self.sqrt_D = torch.nn.Parameter(torch.ones(self.N**3, device=device)) #torch.logspace(-2, 2, self.N**3, device=device) #
        # self.sqrt_D2 = torch.ones(self.shape).cuda()
        # self.sqrt_D2 = torch.nn.Parameter(torch.ones(self.shape, device=device)) #torch.nn.Parameter(torch.logspace(-1, 2, self.N**3, device=device).reshape(self.shape)) #
        self._D2 = None #torch.nn.Parameter(torch.ones(self.shape, device=device))
        self.k_mask = torch.ones(self.N**3, device=device)
        self.k_mask[0] = 0  # to avoid problems with the mean mode

    def G(self, x):
        # x = x * torch.exp(self.logD2)
        x = x * self.D2
        x = hartley(x).flatten(-len(self.shape), -1)
        return x
    
    @property
    def D(self):
        # return torch.exp(self.logD)
        return self.sqrt_D**2 * self.k_mask   # to avoid problems with the mean mode
    
    @property
    def D2(self):
        # return torch.exp(self.logD)
        return F.softplus(self._D2) #torch.sigmoid(self._D2) #self._D2**2  #self.sqrt_D2**2 #torch.sigmoid(self.sqrt_D2) #. TODO remove sigmoid
    
    def G_T(self, x):
        x = hartley(x.unflatten(-1, self.shape))
        x = x * self.D2
        # x = x * torch.exp(self.logD2)
        return x
    
class Precision_Matrix_Masked_FFT_mod(GDG_Factor_Matrix):
    """Parametrization of a symmetric positive definite matrix.
    
    Q = D2 * U_H * D * U_H * D2
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, N, device='cuda'):
        super().__init__()
        self.N = N
        self.shape = 3*(self.N,)
        # self.logD = torch.nn.Parameter(torch.ones(self.N**3))    # tried in the past, sqrt works better usually
        # self.logD2 = torch.nn.Parameter(torch.ones(self.shape))
        self.device = device
        self.sqrt_D = torch.nn.Parameter(torch.ones(self.N**3, device=device))
        self._outer_D2 = torch.nn.Parameter(torch.ones(self.shape, device=device)) #torch.logspace(-2, 2, self.N**3, device=device) #
        # self.sqrt_D2 = torch.ones(self.shape).cuda()
        # self.sqrt_D2 = torch.nn.Parameter(torch.ones(self.shape, device=device)) #torch.nn.Parameter(torch.logspace(-1, 2, self.N**3, device=device).reshape(self.shape)) #
        self._D2 = None #torch.nn.Parameter(torch.ones(self.shape, device=device))
        self.k_mask = torch.ones(self.N**3, device=device)
        self.k_mask[0] = 0  # to avoid problems with the mean mode

    def G(self, x):
        # x = x * torch.exp(self.logD2)
        x = x * self.D2 * self.outer_D2
        x = hartley(x).flatten(-len(self.shape), -1)
        return x
    
    @property
    def D(self):
        # return torch.exp(self.logD)
        return self.sqrt_D**2 * self.k_mask   # to avoid problems with the mean mode
    
    @property
    def outer_D2(self):
        # return torch.exp(self.logD)
        return F.softplus(self._outer_D2) #self._outer_D2**2

    @property
    def D2(self):
        # return torch.exp(self.logD)
        return F.softplus(self._D2) #torch.sigmoid(self._D2) #self._D2**2  #self.sqrt_D2**2 #torch.sigmoid(self.sqrt_D2) #. TODO remove sigmoid
    
    def G_T(self, x):
        x = hartley(x.unflatten(-1, self.shape))
        x = x * self.D2 * self.outer_D2
        # x = x * torch.exp(self.logD2)
        return x
    
class Precision_Matrix_Real(GDG_Factor_Matrix):
    """Parametrization of a symmetric diagonal positive definite matrix diagonal.
    
    Q = D * I
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, N):
        super().__init__()
        self.shape = (N, N, N)
        # self.logD = torch.nn.Parameter(torch.ones(N**3))    # tried in the past, sqrt works better usually
        self.sqrt_D = torch.nn.Parameter(2.6 * torch.ones(N**3).cuda())

    def G(self, x):
        x = x.flatten(-len(self.shape), -1)
        return x
    
    @property
    def D(self):
        # return torch.exp(self.logD)
        return self.sqrt_D**2

    def G_T(self, x):
        x = x.unflatten(-1, self.shape)
        return x

class Precision_Matrix_Real2(GDG_Factor_Matrix):
    """Parametrization of a symmetric diagonal positive definite matrix diagonal.
    
    Q = D * I
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, mask, device='cuda'):
        super().__init__()
        # assert (mask > 0).all(), "All mask values must be positive"
        self.shape = mask.shape    
        self._D = mask.flatten().to(device)

    def G(self, x):
        x = x.flatten(-len(self.shape), -1)
        return x
    
    @property
    def D(self):
        return self._D

    def G_T(self, x):
        x = x.unflatten(-1, self.shape)
        return x

    def apply_inverse(self, x):
        """(B, N, N, N) --> (B, N, N, N)"""
        x = self.G(x)
        x = self.D**-1 * x
        x = self.G_T(x)
        return x

    def sample(self, num_samples=1, whiten=False):
        z = torch.randn(num_samples, *self.D.shape, device=self.D.device)
        # z = self.G(z)
        if not whiten:
            z = z * self.D.detach()**-0.5  # apply sqrt covariance in frequency domain
            z = self.G_T(z)
        return z

class Precision_Matrix_Single_Conv(GDG_Factor_Matrix):
    """Parametrization of a symmetric positive definite matrix.
    
    Q = G_T * D * G
    
    Input and output shapes are (B, N, N, N), where shape = (N, N, N).
    """
    def __init__(self, N, p = 1):
        super().__init__()
        
        self._conv1 = torch.nn.Conv3d(1, 1, 2*p+1, padding = p, bias = False)
        self._conv1.weight = torch.nn.Parameter(torch.ones_like(self._conv1.weight))
        self._conv1T = torch.nn.ConvTranspose3d(1, 1, 2*p+1, padding = p, bias = False)
        # self._logD = torch.nn.Parameter(torch.zeros(N**3)-1.)
        self.sqrt_D = torch.nn.Parameter(torch.ones(N**3))
        self.shape = (N, N, N)

    def G(self, x):
        x = self._conv1(x.unsqueeze(1)).squeeze(1)
        x = x.flatten(1, 3)
        return x
    
    @property
    def D(self):
        # return torch.exp(self.logD)
        return self.sqrt_D**2
    
    def G_T(self, x):
        self._conv1T.weight = self._conv1.weight
        x = x.unflatten(1, self.shape)
        x = self._conv1T(x.unsqueeze(1)).squeeze(1)
        return x

