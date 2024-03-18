import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm
from collections import deque

from numba import njit, prange

'''
A faster implementation of generator.py using numba.

for syntactic clarity:
    n refers to the repetition number of trees - n=1 is a single tree puzzle, n=2 is a double tree puzzle, etc.
    d refers to the dension of the puzzle - d=4 is a 4x4 puzzle, d=5 is a 5x5 puzzle, etc.
    trees refers to a dxd binary array of tree locations
    roots refers to a dxd integer array of tree regions
    forest refers broadly to the puzzle, including both trees and roots. While multiple tree solutions may exist for a given root configuration, 
        a forest can be considered as a single valid solution or as the set of all valid solutions.
'''

# Generation of trees

@njit
def get_conflicting(array, i, j):
    adj_indices = []
    if i > 0 and j > 0:
        adj_indices.append([i-1, j-1])
    if i < array.shape[0] - 1 and j < array.shape[1] - 1:
        adj_indices.append([i+1, j+1])
    if j > 0 and i < array.shape[0] - 1:
        adj_indices.append([i+1, j-1])
    if j < array.shape[1] - 1 and i > 0:
        adj_indices.append([i-1, j+1])
        
    conflicting = np.ones_like(array)
    for adj_index in adj_indices:
        conflicting[adj_index[0], adj_index[1]] = 0
    conflicting[i,:] = 0
    conflicting[:,j] = 0
    return conflicting


@njit
def fast_pad(array, shift = (0,0)):
    col_pad = np.zeros((array.shape[0], 1), dtype=array.dtype)
    row_pad = np.zeros((1, array.shape[1] + 2), dtype=array.dtype)
    if shift[0] == 0:
        array = np.hstack((col_pad, array, col_pad))
    elif shift[0] == -1:
        array = np.hstack((array, col_pad, col_pad))
    elif shift[0] == 1:
        array = np.hstack((col_pad, col_pad, array))
    if shift[1] == 0:
        array = np.vstack((row_pad, array, row_pad))
    elif shift[1] == -1:
        array = np.vstack((array, row_pad, row_pad))
    elif shift[1] == 1:
        array = np.vstack((row_pad, row_pad, array))
    return array


@njit
def fast_adjacents_square(array):
    array = (fast_pad(array, shift=(1,0))[1:-1, 1:-1] + array)
    array = (fast_pad(array, shift=(-1,0))[1:-1, 1:-1] + array)
    array = (fast_pad(array, shift=(0,1))[1:-1, 1:-1] + array)
    array = (fast_pad(array, shift=(0,-1))[1:-1, 1:-1] + array)
    array = np.where(array> 0, 1, 0)
    return array


@njit
def find_available(array, n=1):
    d = array.shape[0]
    cols = np.broadcast_to(np.where(np.sum(array, axis=1) > n-1, 1, 0), (d,d)).T
    rows = np.broadcast_to(np.where(np.sum(array, axis=0) > n-1, 1, 0), (d,d))
    adjs = fast_adjacents_square(array)
    opens = np.where(cols + rows + adjs > 0, 0, 1)
    return opens


@njit
def plant_trees(d, n, maxiter = 1000000):
    trees = np.zeros((d,d), dtype=np.uint8)
    count = 0
    while np.sum(trees) < n*d and count < maxiter:
        avail = find_available(trees, n=n)
        if np.any(avail):
            avail_list = np.argwhere(avail)
            i = np.random.randint(0, avail_list.shape[0])
            trees[avail_list[i][0], avail_list[i][1]] = 1
            continue
        else:
            opens = np.where(trees == 0, 1, 0)
            opens_list = np.argwhere(opens)
            i = np.random.randint(0, opens_list.shape[0])
            
            trees = trees * get_conflicting(trees, opens_list[i][0], opens_list[i][1])
            trees[opens_list[i][0], opens_list[i][1]] = 1
        count += 1

    if count < maxiter:
        print(f'completed in {count} iterations')
    else:
        print(f'exited early after {count} iterations')
    return trees


# Generation of roots

@njit
def fast_pad(array, shift = (0,0)):
    col_pad = np.zeros((array.shape[0], 1), dtype=array.dtype)
    row_pad = np.zeros((1, array.shape[1] + 2), dtype=array.dtype)
    if shift[0] == 0:
        array = np.hstack((col_pad, array, col_pad))
    elif shift[0] == -1:
        array = np.hstack((array, col_pad, col_pad))
    elif shift[0] == 1:
        array = np.hstack((col_pad, col_pad, array))
    if shift[1] == 0:
        array = np.vstack((row_pad, array, row_pad))
    elif shift[1] == -1:
        array = np.vstack((array, row_pad, row_pad))
    elif shift[1] == 1:
        array = np.vstack((row_pad, row_pad, array))
    return array


@njit
def fast_adjacents_cross(array):
    new_array = array.copy()
    new_array = (fast_pad(array, shift=(1,0))[1:-1, 1:-1] + new_array)
    new_array = (fast_pad(array, shift=(-1,0))[1:-1, 1:-1] + new_array)
    new_array = (fast_pad(array, shift=(0,1))[1:-1, 1:-1] + new_array)
    new_array = (fast_pad(array, shift=(0,-1))[1:-1, 1:-1] + new_array)
    new_array = np.where(new_array> 0, 1, 0)
    return new_array


@njit
def get_adj_vals(array, i, j):
    adj_vals = []
    if i > 0:
        adj_vals.append(array[i-1, j])
    if i < array.shape[0] - 1:
        adj_vals.append(array[i+1, j])
    if j > 0:
        adj_vals.append(array[i, j-1])
    if j < array.shape[1] - 1:
        adj_vals.append(array[i, j+1])
    adj_vals = [x for x in adj_vals if x != 0]
    adj_vals = np.unique(np.array(adj_vals))
    return adj_vals


@njit
def get_initial_roots(trees):
    roots = np.zeros_like(trees)
    for i,tree in enumerate(np.argwhere(trees)):
        roots[tree[0], tree[1]] = i+1
    return roots


@njit
def grow_roots(roots, trees):
    adjs = np.where(fast_adjacents_cross(np.where(roots>0,1,0)) * np.where(roots==0, 1, 0)>0, 1, 0)
    adjs_list = np.argwhere(adjs)
    i = np.random.randint(0, len(adjs_list))
    val = get_adj_vals(roots, adjs_list[i][0], adjs_list[i][1])
    roots[adjs_list[i][0], adjs_list[i][1]] = val[np.random.randint(0, val.shape[0])]
    return roots


@njit
def generate_singular_roots(trees):
    roots = get_initial_roots(trees)
    while np.any(np.where(roots == 0, 1, 0)):
        roots = grow_roots(roots, trees)
    return roots    
    

@njit
def adj_mat_from_roots(roots):
    d = roots.shape[0]
    n = np.amax(roots)//d
    adj_mat = np.zeros((d*n, d*n), dtype=roots.dtype)
    
    adj_list = []
    for region in np.unique(roots):
        for index in np.argwhere(roots == region):
            adj_vals = get_adj_vals(roots, index[0], index[1])
            for val in adj_vals:
                adj_list.append((region, val))
    adj_list = list(set(adj_list))
    
    for edge in adj_list:
        adj_mat[edge[0] - 1, edge[1] - 1] = 1
    return adj_mat


@njit
def bfs(adj_mat, start=0):
    visited = [start]
    queue = [start]
    
    while queue:
        m = queue.pop()
        for neighbor in np.argwhere(adj_mat[m] > 0).flatten():
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    if len(visited) == adj_mat.shape[0]:
        return True
    return False


# @njit(parallel=True)
@njit
def calc_merges(adj_mat, n=2, temp=1, maxiter = 1000000):
    done = 0
    count = 0
    merged = []
    ordered_mat = adj_mat.copy()
    d = ordered_mat.shape[0]
    temp_sweep = temp
    indices = np.arange(d, dtype=np.uint8)
        
    while not done and count < maxiter:
        # sorted_mat = np.sum(ordered_mat * np.arange(d, dtype=np.int64), axis=1, dtype=np.int64) / np.sum(ordered_mat, axis=0) + np.random.randn(d) * temp_sweep
        sorted_mat = np.sum(ordered_mat * np.arange(d, dtype=np.float32), axis=1, dtype=np.float32) / np.sum(ordered_mat, axis=0) + np.random.randn(d) * temp_sweep
        ordered_indices = np.argsort(sorted_mat)
        indices = indices[ordered_indices]
        ordered_mat = ordered_mat[ordered_indices,:][:,ordered_indices]
        
        invalid = np.zeros(d * n, dtype=np.uint8)
        # for i in prange(0, d//n):
        for i in range(0, d//n):
            if not bfs(ordered_mat[i*n:i*n+n,i*n:i*n+n]):
                invalid[i] = 1
        count += 1
        # print(invalid)
        if not np.any(invalid):
            done = 1
        if count % 1000 == 0:
            if temp_sweep > n:
                temp_sweep = temp
                ordered_mat = adj_mat.copy()
                indices = np.arange(d, dtype=np.uint8)
            temp_sweep *= 1.01
            
        
    merged = [[indices[i:i+n] + 1] for i in range(0, d, n)]
    if done:
        print(f'merge pattern found after {count} iterations, temp = {int(temp_sweep*1000)}e-3')
    else:
        print(f'exited early after {count} iterations')
    return merged

@njit
def merge_roots(roots, temp=1, maxiter=1000000):
    n = np.amax(roots)//roots.shape[0]
    adj_mat = adj_mat_from_roots(roots)
    merged = calc_merges(adj_mat, n=n, temp=temp, maxiter=maxiter)
    new_roots = np.zeros(roots.shape, dtype=np.int64)
    for i,merge in enumerate(merged):
        for j in range(n):
            new_roots = np.where(roots == merge[0][j], i+1, new_roots)
    return new_roots


# Generating a forest:

def generate_forest(d, n, temp=1, maxiter=1000000):
    trees = plant_trees(d, n, maxiter=maxiter)
    roots = generate_singular_roots(trees)
    roots = merge_roots(roots, temp=temp, maxiter=maxiter)
    forest = np.stack((trees, roots))
    return forest

# Plotting a forest:

def plot_forest(forest, solution = True, cmap='tab20b'):
    trees = forest[0]
    roots = forest[1]
    x = np.linspace(0, roots.shape[0]-1, roots.shape[0])
    y = np.linspace(0, roots.shape[1]-1, roots.shape[1])
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    ax.set_aspect('equal', 'box')
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False)
    ax.pcolormesh(x, y, roots, cmap=cmap, edgecolors='white', linewidths=150/1024/32)
    
    if solution:
        ax.scatter(np.argwhere(trees)[:,1], np.argwhere(trees)[:,0], color='k', marker='*', s=50, zorder=10)