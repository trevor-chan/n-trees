import torch

# Continuous loss function based in Langevin dynamics
class LangevinLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, data):
        rnd_normal = torch.randn([data.shape[0], 1, 1, 1], device=data.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = data
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# Discretized loss function
class EntropyLoss:
    def __init__(self, P_factor=20):
        self.P_factor = P_factor # P_factor test ranges from 10 to 50, hyperparameter controls sampling bias during training

    def __call__(self, net, data):
        rnd_normal = torch.randn([data.shape[0],1,1], device=data.device) * 1
        p = rnd_normal.exp()
        p = (p / self.P_factor).clamp(0, 1) / 2 # clamp flip probability to [0,0.5]
        
        
        weight = 1 / (2 * p) # weight for loss function for balancing preconditioning loss potentially not needed
        y = data
        n = torch.bernoulli(torch.ones_like(y) * (1 - p)) * 2 - 1 # noise equal to a bit flip occuring with probability p
        n = n.to(torch.float32)
        D_yn = net(y * n, p)
        loss = weight * ((D_yn - y) ** 2)
        return loss

# Discretized loss function
class ConditionalEntropyLoss:
    def __init__(self, P_factor=10):
        self.P_factor = P_factor # P_factor test ranges from 10 to 50, hyperparameter controls sampling bias during training

    def __call__(self, net, data, prior):
        rnd_normal = torch.randn([data.shape[0],1,1], device=data.device) * 1
        p = rnd_normal.exp()
        p = (p / self.P_factor).clamp(0, 1) / 2 # clamp flip probability to [0,0.5]
        
        # beta = torch.distributions.beta.Beta(1.3, 4)
        # p = beta.sample([data.shape[0],1,1]).to(device=data.device) / 2
        
        
        weight = 1 / (2 * p) # weight for loss function for balancing preconditioning loss potentially not needed
        y = data
        n = torch.bernoulli(torch.ones_like(y) * (1 - p)) * 2 - 1 # noise equal to a bit flip occuring with probability p
        n = n.to(torch.float32)
        D_yn = net(y * n, prior, p)
        loss = weight * ((D_yn - y) ** 2)
        return loss
    
    # No preconditioning required for the entropy loss function, as inputs and outputs are always scaled to [-1, 1]



# Sample weighting during training...
'''
# sigma sample weighting
    rnd_normal = torch.randn([100000])
    sigma = (rnd_normal * -1.2 + 1.2).exp()
    plt.hist(sigma, bins=500)
    plt.show()
    torch.mean(sigma)/max(sigma)
# p sample weighting, logarithmic
    uniform_rand = torch.rand([100000]) * 7
    p = (uniform_rand * -1).exp()
    plt.hist(p, bins=500)
    plt.show()
    torch.mean(p)
# p sample weighting, log-normal
    rnd_normal = torch.randn([100000]) * 1.2
    p = rnd_normal.exp()
    p = (p / 200).clamp(0, 1)
    plt.hist(p, bins=500)
    plt.show()
    torch.mean(p)
# p sample weighting, clamped log-normal
    rnd_normal = torch.randn([data.shape[0],1,1], device=data.device) * 1
    p = rnd_normal.exp()
    p = (p / self.P_factor).clamp(0, 1) / 2 # clamp flip probability to [0,0.5]
'''