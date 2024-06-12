import numpy as np
import torch
from torch.nn.functional import silu



class LangevinPrecond(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution.
        d,                                  # Number of color channels.
        model,                              # Class name of the underlying model.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.n = n
        self.d = d
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, sigma):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten())
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    
class EntropyPrecond(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution.
        d,                                  # Number of color channels.
        model,                              # Class name of the underlying model.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        p_min           = 0,                # Minimum supported noise level.
        p_max           = 0.5,              # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.n = n
        self.d = d
        self.use_fp16 = use_fp16
        self.p_min = p_min
        self.p_max = p_max
        self.sigma_data = sigma_data
        self.p_data = n/d
        self.model = model

    def forward(self, x, p):
        x = x.to(torch.float32)
        x = x.flatten(start_dim=2)
        p = p.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32        
        c_noise = torch.log(2 * p) / 2 + 1  # empirical scaling factor
        F_x = self.model(x.to(dtype), c_noise.flatten())
        assert F_x.dtype == dtype
        D_x = F_x.to(torch.float32)  # took out the skip connection, possibly important
        D_x = D_x.reshape(-1, self.d + 1, self.d, self.d)
        return D_x
    
    
class ConditionalEntropyPrecond(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution.
        d,                                  # Number of color channels.
        model,                              # Class name of the underlying model.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        p_min           = 0,                # Minimum supported noise level.
        p_max           = 0.5,              # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.n = n
        self.d = d
        self.use_fp16 = use_fp16
        self.p_min = p_min
        self.p_max = p_max
        self.sigma_data = sigma_data
        self.p_data = n/d
        self.model = model

    def forward(self, x, prior, p):
        x = x.to(torch.float32)
        x = x.flatten(start_dim=2)
        
        # -------------------------------------------------------------------------
        # Prior dimensionality should flatten along the sequence dimension, but preserve the onehot encoding dimension
        prior = prior.flatten(start_dim=2)
        # -------------------------------------------------------------------------
        
        p = p.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        # # Why would I need to precondition if it's just -1s and 1s?
        # Still probably need a weighting function for the loss though
        
        c_noise = torch.log(2 * p) / 2 + 1  # empirical scaling factor
        F_x = self.model(x.to(dtype), prior.to(dtype), c_noise.flatten())
        assert F_x.dtype == dtype
        D_x = F_x.to(torch.float32)  # took out the skip connection, possibly important
        D_x = D_x.reshape(-1, 1, self.d, self.d)
        return D_x


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
    
class ConditionalAttention(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution at input/output.
        d,                                  # Number of color channels at input.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        num_heads           = 16,            # Number of layers in the MLP.
        model_dimension     = 512,          # Transformer encoder dimension
        num_encoder_layers  = 6,            # Transformer layers
        embedding_type      = 'fourier', # Timestep embedding type: 'positional' or 'fourier'
        embedding_channels  = 128,          # Number of channels in the timestep embedding.
        activation          = torch.nn.SiLU(),         # Activation function
    ):
        super().__init__()
        
        assert embedding_type in ['fourier', 'positional']
        # init = dict(init_mode='xavier_uniform')
        self.sequence_dimension = int(d * d)
        self.vector_dimension = int(d + 1)
        self.embedding_dimension = embedding_channels
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.activation = activation
        self.model_dimension = model_dimension
        
        #mapping inputs
        self.embed = PositionalEmbedding(num_channels=embedding_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=embedding_channels)
        
        self.map_inputs = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features = self.vector_dimension + self.embedding_dimension, out_features = self.model_dimension),
            torch.nn.LayerNorm(self.model_dimension),
            self.activation,
        )
        
        # --------------------------------------------------------------------------------------------------------------------------------------------------
        # Encoder layer and transformer encoder generally requires a shape of:
        # (batch, sequence_dimension, model_dimension) with memory scaling quadratically in the sequence dimension
        # This equates to (batch, d*d, model_dimension) for a d*d grid, where the model_dimension is roughly on the order of d^3 (d*d*d+d+embedding_dimension)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model_dimension, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # --------------------------------------------------------------------------------------------------------------------------------------------------

        self.map_output = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=self.model_dimension, out_features=self.model_dimension),
            torch.nn.LayerNorm(self.model_dimension),
            self.activation,
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=self.model_dimension, out_features=1),
            torch.nn.Tanh(),
        )

    def forward(self, x, prior, p):        
        emb = self.embed(p).unsqueeze(-1).repeat(1, 1, self.sequence_dimension)
        x = torch.cat((x, prior, emb), dim=1)
        x = x.permute(0, 2, 1)
        x = self.map_inputs(x)
        x = self.transformer_encoder(x)
        x = self.map_output(x)
        x = x.permute(0, 2, 1)
        return x    
    
    
class Attention(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution at input/output.
        d,                                  # Number of color channels at input.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        num_heads           = 8,            # Number of layers in the MLP.
        model_dimension     = 512,          # Transformer encoder dimension
        num_encoder_layers  = 8,            # Transformer layers
        embedding_type      = 'fourier',    # Timestep embedding type: 'positional' or 'fourier'
        embedding_channels  = 128,          # Number of channels in the timestep embedding.
        activation          = torch.nn.SiLU(),         # Activation function
    ):
        super().__init__()
        
        assert embedding_type in ['fourier', 'positional']
        # init = dict(init_mode='xavier_uniform')
        self.sequence_dimension = int(d * d)
        self.vector_dimension = int(d + 1)
        self.embedding_dimension = embedding_channels
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.activation = activation
        self.model_dimension = model_dimension
        
        #mapping inputs
        self.embed = PositionalEmbedding(num_channels=embedding_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=embedding_channels)
        
        self.map_inputs = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features = self.vector_dimension + self.embedding_dimension, out_features = self.model_dimension),
            torch.nn.LayerNorm(self.model_dimension),
            self.activation,
        )
        
        # --------------------------------------------------------------------------------------------------------------------------------------------------
        # Encoder layer and transformer encoder generally requires a shape of:
        # (batch, sequence_dimension, model_dimension) with memory scaling quadratically in the sequence dimension
        # This equates to (batch, d*d, model_dimension) for a d*d grid, where the model_dimension is roughly on the order of d^3 (d*d*d+d+embedding_dimension)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model_dimension, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # --------------------------------------------------------------------------------------------------------------------------------------------------

        self.map_output = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=self.model_dimension, out_features=self.model_dimension),
            torch.nn.LayerNorm(self.model_dimension),
            self.activation,
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=self.model_dimension, out_features=self.vector_dimension),
            torch.nn.Tanh(),
        )

    def forward(self, x, p):
        emb = self.embed(p).unsqueeze(-1).repeat(1, 1, self.sequence_dimension)
        x = torch.cat((x.flatten(start_dim=2), emb), dim=1)
        x = x.permute(0, 2, 1)
        x = self.map_inputs(x)
        x = self.transformer_encoder(x)
        x = self.map_output(x)
        x = x.permute(0, 2, 1)
        return x