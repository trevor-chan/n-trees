import os
import time
import copy
import json
import pickle
import numpy as np
import torch
import datasets


def simple_training_loop(
    run_dir             = '.',      # Output directory.
    dataset             = None,     # Options for training set.
    network             = None,     # Options for model and preconditioning.
    loss                = None,     # Options for loss function.
    optimizer           = None,     # Options for optimizer.
    # conditional         = False,    # Conditional training.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    num_workers         = 4,        # Number of data loading workers.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    device              = torch.device('cuda'),
    kimg_per_tick       = 1000,      # How often to save the training state.
):
    # Initialize.
    start_time = time.time()
    if os.path.isdir(run_dir):
        # raise RuntimeError(f'Output directory "{run_dir}" already exists')
        os.makedirs(run_dir, exist_ok=True)
    else:
        os.makedirs(run_dir, exist_ok=True)

    # Load dataset.
    print('Loading dataset...')
    dataset_obj = dataset
    dataset_sampler = iter(datasets.InfiniteSampler(dataset=dataset_obj, rank=0, num_replicas=1, shuffle=True, seed=seed, window_size=0.5))
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=batch_size, num_workers=num_workers, sampler=dataset_sampler))

    # Construct network.
    print('Constructing network...')
    net = network
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    print('Setting up optimizer...')
    loss_fn = loss
    optimizer = optimizer
    
    # Setup logs.
    print('Setting up logs...')
    running_loss = []
    training_stats = dict(
        tick = [],
        tick_start_time = [],
        tick_runtime = [],
        loss = [],
        std = [],
    )

    # Train.
    print(f'Training for {total_kimg} kimg...')
    print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    # dist.update_progress(cur_nimg // 1000, total_kimg)
    
    while True:
        
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        
        if dataset.conditional:
            data, prior = next(dataset_iterator)
            data = data.to(device).to(torch.float32)
            prior = prior.to(device).to(torch.float32)
            loss = loss_fn(net=net, data=data, prior=prior)
        else:
            data = next(dataset_iterator)
            data = data.to(device).to(torch.float32)
            loss = loss_fn(net=net, data=data)
        loss.sum().backward()
        running_loss.append(loss.cpu())

        # Update weights.
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Perform maintenance tasks once per tick.
        cur_tick += 1
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
            
        # Save full dump of the training state.
        torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats['tick'].append(cur_tick)
        training_stats['tick_start_time'].append(tick_start_time)
        training_stats['tick_runtime'].append(time.time() - tick_start_time)
        training_stats['loss'].append(torch.mean(torch.stack(running_loss)).item())
        training_stats['std'].append(torch.std(torch.stack(running_loss)).item())
        json.dump(training_stats, open(os.path.join(run_dir, 'log.json'), 'wt'))
        running_loss = []
        print(f"tick {cur_tick}, loss {training_stats['loss'][-1]:.4f}, std {training_stats['std'][-1]:.4f}, time {training_stats['tick_runtime'][-1]:.2f} sec")

        # Update state.
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        if done:
            break

    # Done.
    print()
    print('Exiting...')

#----------------------------------------------------------------------------