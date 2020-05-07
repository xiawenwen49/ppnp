import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import ipdb

from .utils import MixedDropout, sparse_matrix_to_torch


def calc_A_hat(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = sp.diags(D_vec_invsqrt_corr)
    return D_invsqrt_corr @ A @ D_invsqrt_corr


def calc_ppr_exact(adj_matrix: sp.spmatrix, alpha: float) -> np.ndarray:
    nnodes = adj_matrix.shape[0]
    M = calc_A_hat(adj_matrix)
    A_inner = sp.eye(nnodes) - (1 - alpha) * M
    return alpha * np.linalg.inv(A_inner.toarray())


class PPRExact(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, drop_prob: float = None):
        super().__init__()

        ppr_mat = calc_ppr_exact(adj_matrix, alpha)
        self.register_buffer('mat', torch.FloatTensor(ppr_mat))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, predictions: torch.FloatTensor, idx: torch.LongTensor):
        return self.dropout(self.mat[idx]) @ predictions


class PPRPowerIteration(nn.Module):
    def __init__(self, adj_matrix: sp.spmatrix, alpha: float, niter: int, drop_prob: float = None):
        super().__init__()
        self.alpha = alpha
        self.niter = niter

        M = calc_A_hat(adj_matrix)
        self.register_buffer('A_hat', sparse_matrix_to_torch((1 - alpha) * M))

        if drop_prob is None or drop_prob == 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)

    def forward(self, local_preds: torch.FloatTensor, idx: torch.LongTensor):
        preds = local_preds
        for _ in range(self.niter):
            A_drop = self.dropout(self.A_hat)
            preds = A_drop @ preds + self.alpha * local_preds
        return preds[idx]


class DiffusionIteration(nn.Module):
    def __init__(self, adj_matrix, niter):
        super().__init__()
        self.niter = niter

        D_mat = calc_D_mat_(adj_matrix, niter)
        self.register_buffer('D_mat', torch.FloatTensor(D_mat))
        self.weight = nn.Parameter(torch.ones((1, niter)), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, local_preds, idx):
        # ipdb.set_trace()
        preds = local_preds
        # weight = self.softmax(self.weight)
        weight = self.weight
        preds = torch.matmul(weight, torch.matmul(self.D_mat, preds).view(self.niter, -1)).view(local_preds.shape)
        # preds = self.weight @ (self.D_mat @ preds)
        # preds = torch.squeeze(preds)
        return preds[idx]

def calc_D_mat_(adj_matrix, niter=5):
    """Calculate the diffusion propagation matrix.
    The j-th column of DI_mat is the iterated influe_vec of seed_vec[0, 0, 0 ..., j-th=1, ..., 0]
    """
    A_hat = calc_A_hat(adj_matrix)
    A_hat = A_hat.toarray() # .T or not?
    N = A_hat.shape[0]
    DI_mat = np.zeros((niter, N, N))
    
    DI_mat[0] = np.eye(N)
    for i in range(1, niter):
        DI_mat[i] = DI_mat[i-1] @ A_hat
        # A_hat = A_hat @ A_hat     # bug
    return DI_mat

def calc_A_hat_(adj_matrix: sp.spmatrix) -> sp.spmatrix:
    nnodes = adj_matrix.shape[0]
    A = adj_matrix + sp.eye(nnodes)
    D_vec = np.sum(A, axis=1).A1
    D_vec_inv_corr = 1 / D_vec
    D_inv_corr = sp.diags(D_vec_inv_corr)
    return D_inv_corr @ A 


class PrePPRIteration(nn.Module):
    def __init__(self, adj_matrix, alpha, niter):
        super().__init__()

        self.niter = niter
        self.alpha = alpha

        A_mat = calc_A_hat(adj_matrix)
        pre_PPR_mat = calc_pre_PPR_mat(A_mat, self.alpha, self.niter)
        self.register_buffer('pre_PPR_mat', torch.FloatTensor(pre_PPR_mat))

    
    def forward(self, local_preds, idx):
        preds = self.pre_PPR_mat @ local_preds
        return preds[idx]

def calc_pre_PPR_mat(A_mat: sp.spmatrix, alpha, niter) -> sp.spmatrix:
    N = A_mat.shape[0]
    pre_PPR_mat = alpha * np.eye(N)
    A_mat_power = A_mat

    for i in range(1, niter):
        pre_PPR_mat = pre_PPR_mat + alpha * (1-alpha)**i * A_mat_power
        A_mat_power = A_mat_power @ A_mat
    
    pre_PPR_mat = pre_PPR_mat + (1-alpha)**niter  * A_mat_power
    return pre_PPR_mat


def calc_D_mat(adj_matrix, niter=5):
    """Calculate the diffusion propagation matrix.
    The j-th column of DI_mat is the iterated influe_vec of seed_vec[0, 0, 0 ..., j-th=1, ..., 0]
    """
    A_hat = calc_A_hat(adj_matrix)
    A_hat = A_hat.toarray()
    N = A_hat.shape[0]
    DI_mat = np.zeros((N, N))

    for i in range(N):
        seed_vec = np.zeros((N, ))
        seed_vec[i] = 1
        seed_idx = np.argwhere( seed_vec == 1)
        iterated = PIteration(A_hat, seed_vec, seed_idx, True, iter=niter)
        DI_mat[:, i] = iterated
    
    return DI_mat


def PIteration(prob_matrix, predictions, seed_idx, substitute=True, iter=3):
    """Final prediction iteration to fit the ideal equation system.
    """
    def one_iter(prob_matrix, predictions):
        P2 = prob_matrix.T @ np.diag(predictions)
        P3 = np.ones(prob_matrix.shape) - P2
        one_iter_preds = np.ones((prob_matrix.shape[0],)) - np.prod(P3, axis=1).flatten()
        return one_iter_preds
    
    predictions = predictions.flatten()
    assert prob_matrix.shape[0] == prob_matrix.shape[1]
    assert predictions.shape[0] == prob_matrix.shape[0]

    import scipy.sparse as sp 
    if sp.isspmatrix(prob_matrix):
        prob_matrix = prob_matrix.toarray()  
            
    # prob_matrix[np.argwhere(np.eye(prob_matrix.shape[0]) == 1)] = 1 # p_ii should = 1
    final_preds = predictions
    for i in range(iter):
        final_preds = one_iter(prob_matrix, final_preds)
        if substitute: # very important !!!
            final_preds[seed_idx] = 1 
    
    return final_preds