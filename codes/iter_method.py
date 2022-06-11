import numpy as np


def gauss_newton_iter_solve(f, X, y, params, f_derrivatives_wrt_params, tol=1e-5, max_iter=1000):

    params_names = params.keys()


    def e_for_params(params):
        y0 = f(X, **params)
        e = y - y0
        return e


    def Z_for_params(params):
        Z = np.zeros((X.shape[0], len(params)))
        for i, param_name in enumerate(params_names):
            df = f_derrivatives_wrt_params[param_name](X, **params)
            Z[:, i] = df
        return Z


    def deltas(Z, e):
        return np.linalg.inv(Z.T @ Z) @ Z.T @ e


    def print_params(params, newline=True):
        print(", ".join([
            f"{param_name}:   {float(params[param_name]):.7}" for param_name in params_names]),
              end = "\n" if newline else "")


    def print_deltas(d, newline = True):
        print(", ".join(
            [f"d_{param_name}: {float(d[i]):.7}" for i, param_name in enumerate(params_names)]),
              end = "\n" if newline else "")

    init = True
    for itr in range(1, max_iter + 1):
        e = e_for_params(params)
        
        if init:
            mse = np.mean(e**2)
            print(f"Initial, MSE: {mse:.5},\nparams:", end="")
            print_params(params)
            init = False
        
        Z = Z_for_params(params)
        d = deltas(Z, e)
        for i, param_name in enumerate(params_names):
            params[param_name] += d[i]
        
        e = e_for_params(params)
        mse = np.mean(e**2)
        print(f"\nIter: {itr},  MSE: {mse:.5}")
        print_deltas(d)
        print_params(params)
        
        if np.all((np.abs(d) < tol)):
            break

    return params
