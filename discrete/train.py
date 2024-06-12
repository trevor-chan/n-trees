import sys
sys.path.append('../n-trees/')
sys.path.append('../utils/')
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datasets
import loss
import models_tests
import training_loop
import utils
import fast_generator

device = torch.device('cuda')

# Create the dataset
# dataset = datasets.ForestDataset(10, 2, temp=1, maxiter=10000, size=1000000)
# torch.save(dataset, '../datasets/dataset_10_2_1e6.pt')
dataset = torch.load('../datasets/dataset_10_2_1e6.pt')

# loss_fn = loss.ConditionalEntropyLoss(P_factor=15)
loss_fn = loss.EntropyLoss(P_factor=15)

# model = models_tests.ConditionalEntropyPrecond(
#     n=dataset.n,
#     d=dataset.d,
#     model =  models_tests.TestingConditionalAttention(dataset.n, 

#                                         dataset.d,                          # dataset dimension
#                                         dropout             = 0.10,         # Dropout probability of intermediate activations.
#                                         num_heads           = 8,            # Number of layers in the MLP.
#                                         model_dimension     = 256,          # Hidden layer size.
#                                         num_encoder_layers  = 4,
#                                         embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
#                                         embedding_channels  = 128,          # Number of channels in the timestep embedding.
#                                     )
# )

model = models_tests.EntropyPrecond(
    n=dataset.n,
    d=dataset.d,
    model =  models_tests.Attention(dataset.n, 
                                    dataset.d,                          # Number of color channels at input.
                                    dropout             = 0.10,         # Dropout probability of intermediate activations.
                                    num_heads           = 16,           # Attention heads
                                    model_dimension     = 512,          # Hidden layer / embedding size.
                                    num_encoder_layers  = 6,            # Number of layers in the transformer MLP.
                                    embedding_type      = 'fourier',    # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                                    embedding_channels  = 128,          # Number of channels in the timestep embedding.
                                    )
).to(device)

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)

# Train
training_loop.simple_training_loop(
    run_dir             = '../experiments/test15_attention_newmodel',      # Output directory.
    dataset             = dataset,      # Options for training set.
    network             = model,        # Options for model and preconditioning.
    loss                = loss_fn,      # Options for loss function.
    optimizer           = optimizer,    # Options for optimizer.
    seed                = 0,            # Global random seed.
    batch_size          = 1024,         # Total batch size for one training iteration.
    num_workers         = 16,           # Number of ndata loading workers.
    total_kimg          = 1024<<11,      # Training duration, measured in thousands of training images.
    device              = device,
    kimg_per_tick       = 1024<<1,        # How often to save the training state.
)
