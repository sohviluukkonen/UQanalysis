import numpy as np
import pandas as pd

from typing import List

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, norm, binned_statistic
from scipy.integrate import simps



def compute_ence_per_target(exp, mean, var, bins=100):

    """
    Compute the expected normalized calibration error (ENCE) for a given set of predictions.
    
    Parameters
    ----------
    exp : array-like
        The experimental values.
    mean : array-like
        The predicted means.
    var : array-like
        The predicted variances.
    bins : int, optional
        The number of bins to use for the calibration curve.
        
    Returns
    -------
    float
        The ENCE.
    """

    nmols = len(exp)
    steps = int(nmols/bins)
    ence = 0
    ranked_confidence_list = np.argsort(var, axis=0).flatten().astype(int)
    for bin in range(bins):
        sel = ranked_confidence_list[bin*steps:(bin+1)*steps]
        rmvar = np.sqrt(np.mean(var[sel]))
        rmse = np.sqrt(mean_squared_error(exp[sel], mean[sel])) 
        ence += np.abs(rmvar-rmse) / rmvar / bins

    return ence

def compute_calibration_curve_per_target(exp, mean, var, step=0.01):

    """
    Compute the calibration curve for a given set of predictions.
    
    Parameters
    ----------
    exp : array-like
        The experimental values.
    mean : array-like
        The predicted means.
    var : array-like
        The predicted variances.
    step : float, optional
        The step size for the calibration curve.
        
    Returns
    -------
    expected_frac : array-like
        The expected fraction of molecules in each bin.
    observed_frac : array-like
        The observed fraction of molecules in each bin.
    mca : float
        The miscalibration area.
    """

    expected_frac, observed_frac = np.array( [ 2*i for i in np.arange(0,0.5+step,step) ] ), []
    nmols = len(exp)

    for j in range(len(expected_frac)):
        plow = 0.5 - expected_frac[j] / 2
        pup = 0.5 + expected_frac[j] / 2
        cum = 0
        for k in range(nmols):
            low = norm.ppf(plow, loc=mean[k], scale=np.sqrt(var[k]))
            up = norm.ppf(pup, loc=mean[k], scale=np.sqrt(var[k]))
            if low <= exp[k] <= up: cum += 1            
        observed_frac.append(cum/nmols)

    mca = simps(np.abs(observed_frac-expected_frac), expected_frac)

    return expected_frac, observed_frac, mca

def compute_rmsedrop_per_target(exp, mean, var):

    """
    Compute the RMSD drop for a given set of predictions.
    
    Parameters
    ----------
    exp : array-like
        The experimental values.
    mean : array-like
        The predicted means.
    var : array-like
        The predicted variances.
        
    Returns
    -------
    rmsedrops : array-like
        The RMSD drops.
    """

    nmols = len(exp)
    ranked_confidence_list = np.argsort(var, axis=0).flatten().astype(int)
    rmse_drop = []
    for k in range(nmols):
        conf = ranked_confidence_list[0:k+1]
        conf_rmse = np.sqrt(mean_squared_error(exp[conf.astype(int)], mean[conf.astype(int)]))
        rmse_drop.append(conf_rmse)

    return np.flip(rmse_drop)

def compute_rmsedrops_curve(targets : List[str], true : pd.DataFrame, pred : pd.DataFrame, exp_sufix: str = None, mean_sufix : str = '_Mean', var_sufix : str = '_Var', bins : int = 100):

    """
    Compute the RMSD drop curve for a given set of predictions for multiple targets

    Parameters
    ----------
    targets : List[str]
        The list of targets.
    true : pd.DataFrame
        The dataframe with true values.
    pred : pd.DataFrame
        The dataframe with predicted values.
    exp_sufix : str, optional
        The sufix for the experimental value columns.
    mean_sufix : str, optional
        The sufix for the predicted mean value columns.
    var_sufix : str, optional
        The sufix for the predicted variance value columns.
    bins : int, optional
        The number of bins to use for the calibration curve.
        
    Returns
    -------
    rmsedrops : list of array-like
        The list of RMSD drops per target.
    mean_rmsedrops : array-like
        The mean RMSD drops.
    lower_rmsedrops : array-like
        The lower limit RMSD drops.
    upper_rmsedrops : array-like
        The upper limit RMSD drops.
    """

    rmsedrops = []

    for target in targets:
        exp = true[target].dropna()
        mean_col = target + mean_sufix if mean_sufix else target
        var_col = target + var_sufix if var_sufix else target
        mean = pred.loc[exp.index, mean_col].to_numpy()
        var = pred.loc[exp.index, var_col].to_numpy()
        exp = exp.to_numpy

        rmsedrops.append(compute_rmsedrop_per_target(exp, mean, var))

    # Bin all curves to the same number of points
    x = np.zeros(len(rmsedrops), bins)
    for i, drop in enumerate(rmsedrops):
        x[i,:] = binned_statistic(np.arange(len(drop)), drop, bins=bins, statistic='mean').statistic

    # Compute the mean and lower/upper bounds
    mean_rmsedrops = np.mean(x, axis=0)
    lower_rmsedrops = np.percentile(x, 2.5, axis=0)
    upper_rmsedrops = np.percentile(x, 97.5, axis=0)

    return rmsedrops, mean_rmsedrops, lower_rmsedrops, upper_rmsedrops

def compute_calibration_curve(targets : List[str], true : pd.DataFrame, pred : pd.DataFrame, exp_sufix: str = None, mean_sufix : str = '_Mean', var_sufix : str = '_Var', step : float = 0.01):

    """
    Compute the calibration curve for a given set of predictions for multiple targets
    
    Parameters
    ----------
    targets : List[str]
        The list of targets.
    true : pd.DataFrame
        The dataframe with true values.
    pred : pd.DataFrame
        The dataframe with predicted values.
    exp_sufix : str, optional
        The sufix for the experimental value columns.
    mean_sufix : str, optional
        The sufix for the predicted mean value columns.
    var_sufix : str, optional
        The sufix for the predicted variance value columns.
    step : float, optional
        The step size for the calibration curve.
        
    Returns
    -------
    expected_frac : array-like
        The expected fraction of molecules in each bin.
    observed_fracs : list of array-like
        The list of observed fraction of molecules in each bin per target.
    mcas : array-like
        The miscalibration areas.
    mean_curve : array-like
        The mean calibration curve.
    lower_curve : array-like
        The lower limit calibration curve.
    upper_curve : array-like
        The upper limit calibration curve.
    """

    observed_fracs, mcas = [], []
    for target in targets:
        exp = true[target].dropna()
        mean_col = target + mean_sufix if mean_sufix else target
        var_col = target + var_sufix if var_sufix else target
        mean = pred.loc[exp.index, mean_col].to_numpy()
        var = pred.loc[exp.index, var_col].to_numpy()
        exp = exp.to_numpy

        expected_frac, observed_frac, mca = compute_calibration_curve_per_target(exp, mean, var, step)
        observed_fracs.append(observed_frac)
        mcas.append(mca)

    # Compute the mean and lower/upper bounds
    mean_curve = np.mean(np.array(observed_fracs), axis=0)
    lower_curve = np.percentile(np.array(observed_fracs), 2.5, axis=0)
    upper_curve = np.percentile(np.array(observed_fracs), 97.5, axis=0)

    return expected_frac, observed_fracs, mcas, mean_curve, lower_curve, upper_curve

