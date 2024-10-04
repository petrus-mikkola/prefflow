
import numpy as np
import torch

def letters_to_indices(letters):
    # Convert a string of letters into a list of indices
    # 'A' corresponds to 0, 'B' to 1, 'C' to 2, etc.
    # Assumes letters are provided in uppercase and are in a valid range.
    return [ord(char) - ord('A') for char in letters]


def convert_to_ranking(X,Y):
    #Inpuformat:
    #X,Y #X.shape = (k,D,N) = (num alternatives,space dimensions, number of rankings)
    #Y is either vector of alphabets representing the order Y.shape = (N,1) or Y is ranking order as indices Y.shape = (N,k)
    #Outputformat:
    #X #X.shape = (k,D,N) = (comp,space dimensions, number of rankings)
    #k = X.shape[0]
    N = X.shape[2]
    #D = X.shape[1]
    newX = X.copy()
    for i in range(N):
        if np.issubdtype(Y.dtype, np.str_) or np.issubdtype(Y.dtype, np.object_):
            rankingorder = letters_to_indices(Y[i].item())
        else:
            rankingorder = Y[i,:]
        newX[:,:,i] = np.take(X[:,:,i], rankingorder, axis=0) #Reorder alternatives to match ranking order
    return newX


def convert_to_ranking_and_change_k(X,Y,k):
    #Inpuformat:
    #X,Y #X.shape = (k,D,N) = (num alternatives,space dimensions, number of rankings)
    #Y is either vector of alphabets representing the order Y.shape = (N,1) or Y is ranking order as indices Y.shape = (N,k)
    #Outputformat:
    #X #X.shape = (k,D,N) = (comp,space dimensions, number of rankings)
    k_original = X.shape[0]
    if k > k_original:
        raise ValueError('Desired k is higher than the original k!')
    N = X.shape[2]
    #D = X.shape[1]
    X = X[:k,:,:]  #Discard alternatives except first k
    newX = X.copy()
    for i in range(N):
        rankingorder = letters_to_indices(Y[i].item())
         #Discard alternatives except first k
        rankingorder = rankingorder[:k]
        sorted_lst = sorted(rankingorder)
        label_map = {num: i for i, num in enumerate(sorted_lst)}
        rankingorder = [label_map[num] for num in rankingorder]
        newX[:,:,i] = np.take(X[:,:,i], rankingorder, axis=0) #Reorder alternatives to match ranking order
    return newX


#Required for "abalone_age"-experiment

def sample_combinations_and_rank(data, n, k, d):
    #input: dataset, n=number of rankings, k=number of alternatives  (and implicitly d-dimensional space)
    #output: n ranked k-wise comparisons as a tensor of shape (k,d,n)
    num_rows, num_cols = data.shape
    X = torch.empty(k,d,0)
    for _ in range(n):
        indices = np.random.choice(num_rows, size=k, replace=False) # Randomly choose k unique indices from the range 0 to num_rows-1
        selected_rows = data[indices,:]
        sorted_rows = selected_rows[selected_rows[:, -1].argsort()[::-1]] #rank based on the last variable "activity" (decreasing order)
        selected_rows = sorted_rows[:,:-1]
        tensor = torch.tensor(selected_rows).unsqueeze(2)
        X = torch.cat((X,tensor),dim=2)
    return X

def empirical_winner_distribution(data,n,noise_level=1e-4): #noise level is essentially zero, but for numerical stability
    samples = []
    for i in range(n):
        noise = np.random.normal(0, noise_level, data.shape[0])
        outputs = data[:, -1] + noise
        argmax_idx = np.argmax(outputs)
        # Find all rows where the column value is equal to the maximum value
        winner = data[argmax_idx, :-1]
        samples.append(winner)
    return np.array(samples)