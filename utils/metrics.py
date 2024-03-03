import numpy as np
from sklearn.metrics import accuracy_score

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

y_cat=22
def metric(pred, true):
    tv, tc = true[0].detach().cpu().numpy(),true[1].detach().cpu().numpy()
    iv, ic = pred[:, :, :-y_cat].detach().cpu().numpy(), pred[:, 0, -y_cat:].detach().cpu().numpy()
    mae = MAE(iv, tv)
    mse = MSE(iv, tv)
    rmse = RMSE(iv, tv)
    mape = MAPE(iv, tv)
    mspe = MSPE(iv, tv)
    accr = accuracy_score(np.argmax(ic,axis=1),tc)
    
    return mae,mse,rmse,mape,mspe,accr