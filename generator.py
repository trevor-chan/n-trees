import matplotlib.pyplot as plt
import numpy as np
import time
import tqdm
from collections import deque
# import multiprocessing as mp

'''
Functions relevant to the generation, display, and brute-force solution of n-tree puzzles

for syntactic clarity:
    n refers to the repetition number of trees - n=1 is a single tree puzzle, n=2 is a double tree puzzle, etc.
    d refers to the dension of the puzzle - d=4 is a 4x4 puzzle, d=5 is a 5x5 puzzle, etc.
    trees refers to a dxd binary array of tree locations
    roots refers to a dxd integer array of tree regions
    forest refers broadly to the puzzle, including both trees and roots. While multiple tree solutions may exist for a given root configuration, 
        a forest can be considered as a single valid solution or as the set of all valid solutions.
'''


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


def adjacents(array, kernel):
    expanded_array = conv2d(array, kernel)
    return expanded_array


def find_adjacents(array):
    kernel = np.ones((3, 3), dtype=int)
    return np.where((adjacents(array, kernel) - array) * array > 0, 1, 0)


def conv2d(a, f):
    a = np.pad(a, ((1,1),(1,1)), mode='constant', constant_values=(0,0))
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)


def find_available(array, n=1):
    kernel = np.ones((3, 3), dtype=int)
    opens_rowcol = array.copy()
    opens_rowcol[np.argwhere(np.where(np.sum(array, axis=1) > n-1, 1, 0)),:] = 1
    opens_rowcol[:,np.argwhere(np.where(np.sum(array, axis=0) > n-1, 1, 0))] = 1
    opens_nonadjacent = np.where(opens_rowcol + adjacents(array, kernel) > 0, 0, 1)
    return opens_nonadjacent


def find_collisions(array, n=1):
    collisions = array.copy()
    collides = np.broadcast_to(np.where(np.sum(array, axis=0) > n, 1, 0), (array.shape[1], array.shape[0])) + np.broadcast_to(np.where(np.sum(array, axis=1) > n, 1, 0), array.shape).T + find_adjacents(array)
    collisions = np.where(collisions * collides > 0, 1, 0)
    return collisions
        
        
def plant_trees(m, n, maxiter = 1000000):
    trees = np.zeros((m,m), dtype=int)
    count = 0
    with tqdm.tqdm(total=maxiter) as pbar:
        while np.sum(trees) < n*m and count < maxiter:
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
            if count % 1000 == 0:
                pbar.update(1000)
            
    pbar.close()
    if count < maxiter:
        print(f'completed in {count} iterations')
        return trees
    else:
        print(f'exited early after {count} iterations')
        return 0
    
    
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


def get_initial_forest(trees):
    forest = np.zeros_like(trees)
    for i,tree in enumerate(np.argwhere(trees)):
        forest[tree[0], tree[1]] = i+1
    return forest

def grow_forest(forest, trees):
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
    adjs = np.where(adjacents(np.where(forest>0,1,0), kernel=kernel) * np.where(forest==0, 1, 0)>0, 1, 0)
    adjs_list = np.argwhere(adjs)
    i = np.random.randint(0, len(adjs_list))
    val = get_adj_vals(forest, adjs_list[i][0], adjs_list[i][1])
    forest[adjs_list[i][0], adjs_list[i][1]] = val[np.random.randint(0, val.shape[0])]
    return forest

def generate_singular_forest(trees):
    forest = get_initial_forest(trees)
    while np.any(np.where(forest == 0, 1, 0)):
        forest = grow_forest(forest, trees)
    return forest

def i_to_index(i, size):
    return (i // size, i % size)

def index_to_i(index, size):
    return index[0] * size + index[1]

def increment_index(index, size):
    if index[1] == size - 1:
        if index[0] == size - 1:
            return None
        return (index[0] + 1, 0)
    else:
        return (index[0], index[1] + 1)
    
def find_available_forest(trees, forest, visited):
    kernel = np.ones((3, 3), dtype=int)
    opens = trees.copy()
    opens[np.argwhere(np.where(np.sum(trees, axis=1) > 0, 1, 0)),:] = 1
    opens[:,np.argwhere(np.where(np.sum(trees, axis=0) > 0, 1, 0))] = 1
    opens = np.where(opens + adjacents(trees, kernel) + visited > 0, 0, 1)
    for new_idx in np.argwhere(opens): # check if there is already a tree in the region
        region = forest[new_idx[0], new_idx[1]]
        if np.any(trees * np.where(forest == region, 1, 0)):
            opens[new_idx[0], new_idx[1]] = 0
    return opens
        
        
def place_tree(forest, visited=None, trees_pred=None, solutions=None):
    if trees_pred is None:
        trees_pred = np.zeros_like(forest)
    if solutions is None:
        solutions = []
    if visited is None:
        visited = np.zeros_like(forest)
    
    avail = find_available_forest(trees_pred, forest, visited)
    if np.any(avail):
        for avail_index in np.argwhere(avail):
            new_trees_pred = trees_pred.copy()
            new_trees_pred[avail_index[0], avail_index[1]] = 1
            result = place_tree(forest, visited, new_trees_pred, solutions)
            if result is not None:
                solutions = result[0]
                visited = result[1]
        if len(solutions) > 0:
            return solutions, visited
        else:
            return None
    else:
        if np.sum(trees_pred) == forest.shape[0]:
            solutions.append(trees_pred)
            visited = np.where(visited + trees_pred > 0, 1, 0)
            # print('solution found')
            return solutions, visited
        else:
            return None


def calc_adjacency_list(forest):
    adj_list = []
    for region in np.unique(forest):
        for index in np.argwhere(forest == region):
            adj_vals = get_adj_vals(forest, index[0], index[1])
            for val in adj_vals:
                adj_list.append((region, val))
    adj_list = list(set(adj_list))
    adj_list = sorted(adj_list, key=lambda x: x[0] * forest.shape[0] + x[1])
    return adj_list


def adj_list_to_matrix(adjacency_list):
    adj_matrix = np.zeros((adjacency_list[-1][0], adjacency_list[-1][1]))
    for edge in adjacency_list:
        adj_matrix[edge[0] - 1, edge[1] - 1] = 1
    return adj_matrix


def bfs(adj_mat, start=0):
    visited = deque([start])
    queue = deque([start])
    
    while queue:
        m = queue.popleft()
        for neighbor in np.argwhere(adj_mat[m] > 0).flatten():
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
    if len(visited) == adj_mat.shape[0]:
        return True
    return False


def calc_merges(adjacency_list, n=2, maxiter = 1000000, temp=0.25):
    done = 0
    count = 0
    merged = []
    ordered_mat = adj_list_to_matrix(adjacency_list)
    plt.show()
    d = ordered_mat.shape[0]
    temp_sweep = temp
    indices = np.arange(d)
        
    with tqdm.tqdm(total=maxiter) as pbar:
        while not done and count < maxiter:
            ordered_indices = np.argsort(np.sum(ordered_mat * np.arange(d), axis=1) / np.sum(ordered_mat, axis=0) + np.random.randn(d) * temp_sweep)
            indices = indices[ordered_indices]
            ordered_mat = ordered_mat[ordered_indices,:][:,ordered_indices]
            
            valid = 1
            for i in range(0, d, n):
                if not bfs(ordered_mat[i:i+n,i:i+n]):
                    count += 1
                    if count % 1000 == 0:
                        if temp_sweep > n:
                            temp_sweep = temp
                            ordered_mat = adj_list_to_matrix(adjacency_list)
                            indices = np.arange(d)
                        temp_sweep = temp_sweep * 1.01
                        pbar.update(1000)
                    valid = 0
                    break
            if valid:
                done = 1
            
    pbar.close()
        
    merged = [tuple(indices[i:i+n] + 1) for i in range(0, d, n)]
    plt.imshow(ordered_mat)
    if done:
        print(f'merge pattern found after {count} iterations, temp = {temp_sweep}')
    else:
        print(f'exited early after {count} iterations')
    return merged


def merge_forest(forest, adjacency_list, temp=0.25, maxiter=1000000):
    n = np.amax(forest)//forest.shape[0]
    merged = calc_merges(adjacency_list, n=n, temp=temp, maxiter=maxiter)
    if merged == 0:
        return 0
    new_forest = np.zeros_like(forest)
    for i,merge in enumerate(merged):
        for j in range(n):
            new_forest = np.where(forest == merge[j], i+1, new_forest)
    # print(merged)
    return new_forest


# def gr(trees, workers, maxiter=1000000):
#     # for a given tree configuration, start w workers
#     # for each worker, generate a forest, then look for a merge
    






def plot_forest(forest, trees, solution = False, cmap='tab20b'):
    x = np.linspace(0, forest.shape[0]-1, forest.shape[0])
    y = np.linspace(0, forest.shape[1]-1, forest.shape[1])
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    ax.set_aspect('equal', 'box')
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False)
    ax.pcolormesh(x, y, forest, cmap=cmap, edgecolors='white', linewidths=150/1024/32)
    
    if solution:
        ax.scatter(np.argwhere(trees)[:,1], np.argwhere(trees)[:,0], color='k', marker='*', s=50, zorder=10)