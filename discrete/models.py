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
        x = x.flatten(start_dim=1)
        p = p.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        # # Why would I need to precondition if it's just -1s and 1s?
        # Still probably need a weighting function for the loss though
        
        c_noise = torch.log(2 * p) / 2 + 1  # empirical scaling factor
        F_x = self.model(x.to(dtype), c_noise.flatten())
        assert F_x.dtype == dtype
        D_x = F_x.to(torch.float32)  # took out the skip connection, possibly important
        D_x = D_x.reshape(-1, self.d, self.d)
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
        x = x.flatten(start_dim=1)
        prior = prior.flatten(start_dim=1)
        p = p.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32
        
        # # Why would I need to precondition if it's just -1s and 1s?
        # Still probably need a weighting function for the loss though
        
        c_noise = torch.log(2 * p) / 2 + 1  # empirical scaling factor
        F_x = self.model(x.to(dtype), prior.to(dtype), c_noise.flatten())
        assert F_x.dtype == dtype
        D_x = F_x.to(torch.float32)  # took out the skip connection, possibly important
        D_x = D_x.reshape(-1, self.d, self.d)
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
    
    
    
class MLP(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution at input/output.
        d,                                  # Number of color channels at input.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        num_hidden          = 4,            # Number of layers in the MLP.
        hidden_size         = 256,          # Hidden layer size.
        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        embedding_channels  = 128,          # Number of channels in the timestep embedding.
        activation          = silu,         # Activation function
    ):
        assert embedding_type in ['fourier', 'positional']

        super().__init__()
        # init = dict(init_mode='xavier_uniform')
        
        # input_dimension = int(d * d + embedding_channels)
        self.input_dimension = int(d * d + embedding_channels)
        self.output_dimension = int(d * d)
      
        self.embed = PositionalEmbedding(num_channels=embedding_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=embedding_channels)
        self.map_input = torch.nn.Linear(in_features=self.input_dimension, out_features=hidden_size)
        self.num_hidden = num_hidden
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(num_hidden):
            self.hidden_layers.append(torch.nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.map_output = torch.nn.Linear(in_features=hidden_size, out_features=self.output_dimension)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout) 
        

    def forward(self, x, p):
        emb = self.embed(p)
        x = self.activation(self.map_input(torch.cat((x, emb), dim=1)))
        for i in range(self.num_hidden):
            x = self.activation(self.dropout(self.hidden_layers[i](x)))
        x = torch.nn.functional.tanh(self.map_output(x))
        x = x.reshape((-1,self.output_dimension))
        return x


class ConditionalMLP(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution at input/output.
        d,                                  # Number of color channels at input.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        num_hidden          = 4,            # Number of layers in the MLP.
        hidden_size         = 256,          # Hidden layer size.
        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        embedding_channels  = 128,          # Number of channels in the timestep embedding.
        activation          = silu,         # Activation function
    ):
        assert embedding_type in ['fourier', 'positional']

        super().__init__()
        # init = dict(init_mode='xavier_uniform')
        
        self.x_dimension = int(d * d)
        self.prior_dimension = int(d * d * d)
        self.embedding_dimension = embedding_channels
        
        if self.x_dimension > hidden_size * 2:
            print(f'Warning: the hidden size of the network {hidden_size} is much smaller than the data dimension {self.x_dimension}. Consider increasing the hidden size.')
        
        #mapping inputs
        self.embed = PositionalEmbedding(num_channels=embedding_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=embedding_channels)        
        
        # self.map_embedding = torch.nn.Linear(in_features=self.embedding_dimension, out_features=self.embedding_dimension)
        # self.map_prior = torch.nn.Linear(in_features=self.prior_dimension, out_features=self.x_dimension)
        # self.map_x = torch.nn.Linear(in_features=self.x_dimension, out_features=hidden_size)
        
        # self.map_inputs = torch.nn.Linear(in_features = 2 * self.x_dimension + self.embedding_dimension, out_features = hidden_size)
        self.map_inputs = torch.nn.Linear(in_features = self.x_dimension + self.embedding_dimension + self.prior_dimension, out_features = hidden_size)
        print(f'mapping input using: {self.x_dimension + self.embedding_dimension + self.prior_dimension}')
        
        self.num_hidden = num_hidden
        self.hidden_layers = torch.nn.ModuleList()
        for i in range(num_hidden):
            self.hidden_layers.append(torch.nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.map_output = torch.nn.Linear(in_features=hidden_size, out_features=self.x_dimension)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, prior, p):
        emb = self.embed(p)
        # emb = self.activation(self.map_embedding(emb))
        # prior = self.activation(self.map_prior(prior))
        # x = self.activation(self.map_x(x))
        
        x = self.activation(self.map_inputs(torch.cat((x, prior, emb), dim=1)))
        for i in range(self.num_hidden):
            x = self.activation(self.dropout(self.hidden_layers[i](x)))
        x = torch.nn.functional.tanh(self.map_output(x))
        x = x.reshape((-1, self.x_dimension))
        return x
    
    
class Attention(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution at input/output.
        d,                                  # Number of color channels at input.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        num_heads           = 8,            # Number of layers in the MLP.
        model_dimension     = 512,          # Hidden layer size.
        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        embedding_channels  = 128,          # Number of channels in the timestep embedding.
        activation          = silu,         # Activation function
    ):
        assert embedding_type in ['fourier', 'positional']

        super().__init__()
        # init = dict(init_mode='xavier_uniform')
        
        self.input_dimension = int(d * d * (d + 1) + embedding_channels)
        self.output_dimension = int(d * d)
      
        self.embed = PositionalEmbedding(num_channels=embedding_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=embedding_channels)
        self.map_input = torch.nn.Linear(in_features=self.input_dimension, out_features=model_dimension)
        self.map_output = torch.nn.Linear(in_features=model_dimension, out_features=self.output_dimension)
        self.activation = activation
        self.transformer = torch.nn.Transformer(d_model=model_dimension, 
                                                nhead=num_heads, 
                                                num_encoder_layers=6, 
                                                num_decoder_layers=6, 
                                                dim_feedforward=2048, 
                                                dropout=dropout, 
                                                layer_norm_eps=1e-05, 
                                                batch_first=False, 
                                                norm_first=False, 
                                                bias=True)

    def forward(self, x, p):
        emb = self.embed(p)
        x = x.flatten()
        x = self.activation(self.map_input(torch.cat((x, emb))))
        
        x = self.transformer(x)
        
        x = torch.nn.Tanh(self.map_output(x))
        x = x.reshape(self.output_dimension)
        return x
    
    
class ConditionalAttention(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution at input/output.
        d,                                  # Number of color channels at input.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        num_heads           = 8,            # Number of layers in the MLP.
        model_dimension     = 1024,          # Hidden layer size.
        num_encoder_layers  = 8,
        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        embedding_channels  = 128,          # Number of channels in the timestep embedding.
        activation          = torch.nn.SiLU(),         # Activation function
    ):
        assert embedding_type in ['fourier', 'positional']

        super().__init__()
        # init = dict(init_mode='xavier_uniform')
        
        self.x_dimension = int(d * d)
        self.prior_dimension = int(d * d * d)
        self.embedding_dimension = embedding_channels
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.activation = activation
        
        if self.x_dimension > model_dimension * 2:
            print(f'Warning: the hidden size of the network {model_dimension} is much smaller than the data dimension {self.x_dimension}. Consider increasing the hidden size.')
        
        #mapping inputs
        self.embed = PositionalEmbedding(num_channels=embedding_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=embedding_channels)
        
        self.map_inputs = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features = self.x_dimension + self.prior_dimension + self.embedding_dimension, out_features = model_dimension),
        )
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_dimension, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.map_output = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=model_dimension, out_features=model_dimension),
            torch.nn.LayerNorm(model_dimension),
            self.activation,
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=model_dimension, out_features=model_dimension),
            torch.nn.LayerNorm(model_dimension),
            self.activation,
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=model_dimension, out_features=self.x_dimension),
            torch.nn.Tanh(),
        )

    def forward(self, x, prior, p):
        emb = self.embed(p)
        # emb = self.activation(self.map_embedding(emb))
        # prior = self.activation(self.map_prior(prior))
        
        x = self.activation(self.map_inputs(torch.cat((x, prior, emb), dim=1)))
        x = self.transformer_encoder(x)
        x = self.map_output(x)
        x = x.reshape((-1, self.x_dimension))
        return x


class TestingConditionalAttention(torch.nn.Module):
    def __init__(self,
        n,                                  # Image resolution at input/output.
        d,                                  # Number of color channels at input.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        num_heads           = 8,            # Number of layers in the MLP.
        model_dimension     = 1024,         # Hidden layer size.
        num_encoder_layers  = 8,
        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        embedding_channels  = 128,          # Number of channels in the timestep embedding.
        activation          = torch.nn.SiLU(),         # Activation function
    ):
        assert embedding_type in ['fourier', 'positional']

        super().__init__()
        # init = dict(init_mode='xavier_uniform')
        
        self.x_dimension = int(d * d)
        self.prior_dimension = int(d * d * d)
        self.embedding_dimension = embedding_channels
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.activation = activation
        
        if self.x_dimension > model_dimension * 2:
            print(f'Warning: the hidden size of the network {model_dimension} is much smaller than the data dimension {self.x_dimension}. Consider increasing the hidden size.')
        
        #mapping inputs
        self.embed = PositionalEmbedding(num_channels=embedding_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=embedding_channels)
        
        self.map_inputs = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features = self.x_dimension + self.prior_dimension + self.embedding_dimension, out_features = model_dimension),
        )
        
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=model_dimension, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.map_output = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=model_dimension, out_features=model_dimension),
            torch.nn.LayerNorm(model_dimension),
            self.activation,
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=model_dimension, out_features=model_dimension),
            torch.nn.LayerNorm(model_dimension),
            self.activation,
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(in_features=model_dimension, out_features=self.x_dimension),
            torch.nn.Tanh(),
        )

    def forward(self, x, prior, p):
        emb = self.embed(p)
        # emb = self.activation(self.map_embedding(emb))
        # prior = self.activation(self.map_prior(prior))
        
        x = self.activation(self.map_inputs(torch.cat((x, prior, emb), dim=1)))
        x = self.transformer_encoder(x)
        x = self.map_output(x)
        x = x.reshape((-1, self.x_dimension))
        return x
    

# class ConditionalAttentionSimple(torch.nn.Module):
#     def __init__(self,
#         n,                                  # Image resolution at input/output.
#         d,                                  # Number of color channels at input.
#         dropout             = 0.10,         # Dropout probability of intermediate activations.
#         num_heads           = 12,            # Number of self attention heads
#         num_encoder_layers  = 8,            # Number of layers in the MLP.
#         embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
#         embedding_channels  = 128,          # Number of channels in the timestep embedding.
#     ):
#         assert embedding_type in ['fourier', 'positional']

#         super().__init__()
#         # init = dict(init_mode='xavier_uniform')
        
#         self.x_dimension = int(d * d)
#         self.prior_dimension = int(d * d * d)
#         self.embedding_dimension = embedding_channels
#         self.num_encoder_layers = num_encoder_layers
#         self.model_dimension = self.x_dimension + self.prior_dimension + self.embedding_dimension
        
#         if self.x_dimension > self.model_dimension * 2:
#             print(f'Warning: the hidden size of the network {self.model_dimension} is much smaller than the data dimension {self.x_dimension}. Consider increasing the hidden size.')
        
#         #mapping inputs
#         self.embed = PositionalEmbedding(num_channels=embedding_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=embedding_channels)
        
#         encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.model_dimension, nhead=num_heads, dropout=dropout)
#         self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

#         self.map_output = torch.nn.Linear(in_features=self.model_dimension, out_features=self.x_dimension)

#     def forward(self, x, prior, p):
#         emb = self.embed(p)
        
#         x = torch.cat((x, prior, emb), dim=1)
#         x = self.transformer_encoder(x)
#         x = torch.nn.functional.tanh(self.map_output(x))
#         x = x.reshape((-1, self.x_dimension))
#         return x




