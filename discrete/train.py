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

loss_fn = loss.ConditionalEntropyLoss(P_factor=15)


model = models_tests.ConditionalEntropyPrecond(
    n=dataset.n,
    d=dataset.d,
    model =  models_tests.TestingConditionalAttention(dataset.n, 
                                        dataset.d,                                # Number of color channels at input.
                                        dropout             = 0.10,         # Dropout probability of intermediate activations.
                                        num_heads           = 8,            # Number of layers in the MLP.
                                        model_dimension     = 256,          # Hidden layer size.
                                        num_encoder_layers  = 4,
                                        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                                        embedding_channels  = 128,          # Number of channels in the timestep embedding.
                                    )
)

# Create the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.005)


# Train
training_loop.simple_training_loop(
    run_dir             = '../experiments/test12_attention',      # Output directory.
    dataset             = dataset,      # Options for training set.
    network             = model,        # Options for model and preconditioning.
    loss                = loss_fn,      # Options for loss function.
    optimizer           = optimizer,    # Options for optimizer.
    seed                = 0,            # Global random seed.
    batch_size          = 128,          # Total batch size for one training iteration.
    num_workers         = 16,           # Number of ndata loading workers.
    total_kimg          = 2000000,      # Training duration, measured in thousands of training images.
    device              = device,
    kimg_per_tick       = 10000,        # How often to save the training state.
)