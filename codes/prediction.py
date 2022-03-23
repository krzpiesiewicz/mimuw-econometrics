import numpy as np
import pandas as pd


class PredictionModel:
    def __init__(self, log=False, intercept=False):
        self.log = log
        self.intercept = intercept
        
    def fit(self, X, Y, intv=None):
        if intv is not None:
            X = intv.view(X)
            Y = intv.view(Y)
        if self.log:
            X = np.log(X)
            Y = np.log(Y)
        if self.intercept:
            X = X.assign(intercept=1)
        self.__fit__(X, Y)
        self.params = self.params.set_index(X.columns)
        self.Y_columns = Y.columns
        return self
    
    def predict(self, X):
        if self.log:
            X = np.log(X)
        if self.intercept:
            X = X.assign(intercept=1)
        Y = self.__predict__(X)
        if self.log:
            Y = np.exp(Y)
        Y.columns = self.Y_columns
        return Y
    

class OLS(PredictionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __fit__(self, X, Y):
        self.params = np.linalg.inv(X.T @ X) @ X.T @ Y  # Beta
    
    def __predict__(self, X):
        Y = X @ self.params
        return Y


def RSS(Y_true, Y_pred):
    return Y_true.T @ Y_pred
    
    
def evaluate(model, X, Y, steps, eval_intv, score, jump=1):
    begin = eval_intv.begin
    end = eval_intv.shifted_idx(begin, shift=steps)
    scores = []
    
    while end <= eval_intv.index()[-1] and begin < end:
        forecast_intv = eval_intv(begin=begin, end=end)
        model.fit(forecast_intv.prev_view(X), forecast_intv.prev_view(Y))
        Y_pred = model.predict(forecast_intv.view(X))
        scores.append(score(Y_pred, forecast_intv.view(Y)))
        begin = eval_intv.shifted_idx(begin, shift=jump)
        end = eval_intv.shifted_idx(begin, shift=steps)
        
    return np.mean(scores)

def get_one_step_predictions(model, X, Y, forecast_intv):
    model.fit(forecast_intv.prev_view(X), forecast_intv.prev_view(Y))
    preds = model.predict(forecast_intv.view(X))
    for idx in preds.index:
        forecast_intv = forecast_intv(begin=idx, end=idx, nexts=1)
        model.fit(X, Y, forecast_intv.prev())
        preds.loc[idx:idx,:] = model.predict(X.loc[idx:idx,:])
    return preds
