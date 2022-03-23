import numpy as np
import pandas as pd
import itertools

from .statistics import r


def __get_X_Y_L__(X, Y=None):
    if type(X) is not pd.DataFrame:
        X = pd.DataFrame(X)
    if Y is None:
        Y = X.iloc[:,0]
        X = X.iloc[:,1:]
    else:
        if type(Y) is not pd.DataFrame:
            Y = pd.DataFrame(Y)
    L = X.shape[1]
    return X, Y, L


def __get_r0_R__(r0=None, R=None, X=None, Y=None):
    # Only one of the pairs should be provided: (r0, R) or (X, Y).
    
    if r0 is not None or R is not None:
        assert r0 is not None and R is not None
        if type(r0) is not pd.DataFrame:
            r0 = pd.DataFrame(r0)
        if type(R) is not pd.DataFrame:
            r0 = pd.DataFrame(r0)
    else:
        assert X is not None
        X, Y, _ = __get_X_Y_L__(X, Y)
        if r0 is None:
            r0 = hellwig_r0(X, Y)
        if R is None:
            R = hellwig_R(X, Y)
    return r0, R


def __get__J_cols__(columns, J_idx=None, J_cols=None, J_mask=None):
    # Only one of J_idx, J_cols, and J_mask should be provided.
    
    assert sum([J is not None for J in [J_idx, J_cols, J_mask]]) == 1
    
    if J_mask is not None:
        J_idx = np.where(J_mask)
    
    if J_idx is not None:
        if type(J_idx) is not pd.Index:
            J_idx = pd.Index(J_idx)
        J_cols = columns[tuple(J_idx,)]
        
    if type(J_cols) is not pd.Index:
        J_cols = pd.Index(J_cols)
    return J_cols
    

def hellwig_r0(X, Y=None):
    X, Y, L = __get_X_Y_L__(X, Y)
    r0 = np.zeros(L)
    for j in range(L):
        r0[j] = r(Y, X.iloc[:,j])
    r0 = pd.DataFrame(r0)
    r0.columns = ["r0"]
    return r0.set_index(pd.Index(X.columns))


def hellwig_R(X, Y=None):
    X, Y, L = __get_X_Y_L__(X, Y)
    R = np.eye(L)
    for i in range(L):
        for j in range(L):
            if i != j:
                R[i,j] = r(X.iloc[:,i], X.iloc[:,j])
    R = pd.DataFrame(R)
    R.columns = X.columns
    return R.set_index(pd.Index(X.columns))
    return R


def hellwig_h(r0=None, R=None, X=None, Y=None, J_idx=None, J_cols=None, J_mask=None, return_H=True):
    # r0 and R - respectively: a vector and a matrix from Hellwig's method,
    # X - a matrix with a dependent X[,0] and independent X[,1:].
    # Only one of the pairs should be provided: (r0, R) or (X, Y).
    #
    # J_idx - columns numbers (from 0 to m-1),
    # J_cols - columns names,
    # J_mask - 0-1 mask of columns.
    # Only one of J_idx and J_cols should be provided.
    
    r0, R = __get_r0_R__(r0, R, X, Y)
        
    J_cols = __get__J_cols__(R.columns, J_idx, J_cols, J_mask)
        
    R = R.loc[J_cols, J_cols]
    r0 = r0.loc[J_cols]
    r0.columns = [""]
    
    R_sum = pd.DataFrame(np.sum(np.abs(R), axis=0))
    R_sum.columns = [""]
    
    h = np.square(r0) / R_sum
    h.columns = ["h"]
    
    if return_H:
        H = float(np.sum(h))
        return (h, H)
    return h


def helwig_subset(r0=None, R=None, X=None, Y=None, return_all_subsets=False, return_as_comb=False):
    # r0 and R - respectively: a vector and a matrix from Hellwig's method,
    # X - a matrix with a dependent X[,0] and independent X[,1:].
    # Only one of the pairs should be provided: (r0, R) or (X, Y).
    #
    # J_idx - columns numbers,
    # J_cols = columns names.
    # Only one of J_idx and J_cols should be provided.
    
    r0, R = __get_r0_R__(r0, R, X, Y)
    
    subsets_res = []
    
    is_first = True
    for comb in itertools.product([0, 1], repeat = len(r0)):
        if is_first:
            is_first = False
            continue
        h, H = hellwig_h(r0=r0, R=R, J_mask=comb, return_H=True)
        subset = comb if return_as_comb else R.columns[np.where(comb)]
        subsets_res.append({"H": H, "h": h, "subset": subset})
    
    subset_res = sorted(subsets_res, reverse=True, key=lambda d: d["H"])
    
    if return_all_subsets:
        return subset_res
    else:
        return subset_res[0]
