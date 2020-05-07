from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 

from .utils import MixedLinear, MixedDropout
from .propagation import calc_A_hat, calc_pre_PPR_mat

class PPNP(nn.Module):
    def __init__(self, nfeatures: int, nclasses: int, hiddenunits: List[int], drop_prob: float,
                 propagation: nn.Module, bias: bool = False):
        super().__init__()

        fcs = [MixedLinear(nfeatures, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i], bias=bias))
        fcs.append(nn.Linear(hiddenunits[-1], nclasses, bias=bias))
        self.fcs = nn.ModuleList(fcs)

        self.reg_params = list(self.fcs[0].parameters())

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)
        self.act_fn = nn.ReLU()

        self.propagation = propagation

    def _transform_features(self, attr_matrix: torch.sparse.FloatTensor):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_matrix)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.fcs[-1](self.dropout(layer_inner))
        return res

    def forward(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor):
        local_logits = self._transform_features(attr_matrix)
        final_logits = self.propagation(local_logits, idx)
        return F.log_softmax(final_logits, dim=-1) # log probabilities


class PPRGCN(nn.Module):
    def __init__(self, nfeatures: int, nclasses: int, adj_matrix: sp.spmatrix, hiddenunits: List[int], drop_prob: float, niter: int, alpha: float, bias: bool = False):
        super().__init__()

        fcs = [MixedLinear(nfeatures, hiddenunits[0], bias=bias)]
        for i in range(1, len(hiddenunits)):
            fcs.append(nn.Linear(hiddenunits[i - 1], hiddenunits[i], bias=bias))
        fcs.append(nn.Linear(hiddenunits[-1] *2, nclasses, bias=bias))
        self.fcs = nn.ModuleList(fcs)

        self.reg_params = list(self.fcs[0].parameters())

        if drop_prob is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(drop_prob)
        self.act_fn = nn.ReLU()

        # self.propagation = propagation

        self.niter = niter
        self.alpha = alpha
        A_mat = calc_A_hat(adj_matrix)
        pre_PPR_mat = calc_pre_PPR_mat(A_mat, self.alpha, self.niter)
        self.register_buffer('pre_PPR_mat', torch.FloatTensor(pre_PPR_mat))

    def _transform_features(self, attr_matrix: torch.sparse.FloatTensor):
        # attr_matrix = attr_matrix.to_dense()
        # prop_mat = self.pre_PPR_mat @ attr_matrix

        layer_inner = self.act_fn(self.fcs[0](self.dropout(attr_matrix )))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc( layer_inner ))
        
        prop_mat = self.pre_PPR_mat @ layer_inner
        layer_inner = torch.cat((prop_mat, layer_inner), dim=1)
        res = self.fcs[-1](self.dropout(layer_inner))

        res = self.pre_PPR_mat @ res

        return res

    def forward(self, attr_matrix: torch.sparse.FloatTensor, idx: torch.LongTensor):
        local_logits = self._transform_features(attr_matrix)
        # final_logits = self.propagation(local_logits, idx)
        final_logits = local_logits[idx]
        return F.log_softmax(final_logits, dim=-1) # log probabilities
    

# class GCN(nn.Module):
#     def __init__(self):
#         pass

# class GAT(nn.Module):
#     def __init__(self):
#         pass

# class SGC(nn.Module):
#     def __init__(self):
#         pass
