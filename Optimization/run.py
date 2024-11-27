import os
import shutil
import numpy as np
import pandas as pd
from glob import glob

# Imports from PET
from popt.loop.ensemble import Ensemble
from popt.update_schemes.linesearch import LineSearch
from popt.misc_tools.optim_tools import time_correlation, corr2cov
from simulator.opm import flow

# Local import
from objective import function

#---------------------------------------------------------------------------------------------
wwir_max = 8000     # Sm3/day
wwir_min = 0        # Sm3/day
fopr_max = 18000    # Sm3/day
fopr_min = 0        # Sm3/day

# Initial states
wwir_init = 108*[6000]
fopr_init = 'init_fopr.npz'

# Initial variance
var_inj  = 0.0025*wwir_max**2  # σ = 5%
var_prod = 0.0025*fopr_max**2  # σ = 5%

# Prior states
prior_injrate  = [['mean', wwir_init],
                  ['var', var_inj],
                  ['limits', wwir_min, wwir_max]]

prior_prodrate = [['mean', fopr_init],
                  ['var', var_prod],
                  ['limits', fopr_min, fopr_max]]

# NPV parameters
npv_params = [['wop', 600.0], # oil price [USD/Sm³]
              ['wgp', 17.0],  # gas price [USD/Sm³]
              ['wwp', 38.0],  # cost of water disposal  [USD/Sm³]
              ['wwi', 18.0],  # cost of water injection [USD/Sm³]
              ['wem', 150],   # co2 tax [USD/ton]
              ['disc', 0.08]] # discount factor

# Report dates for production data
report_dates = pd.date_range(start='2020-08-01', 
                             end='2025-01-01', 
                             freq='MS').to_pydatetime().tolist()

# Ensemble settings
kwargs_ens = {'ne': 50,
              'state': ['injrate', 'prodrate'],
              'transform': True,
              'num_models': 50,
              'prior_injrate': prior_injrate,
              'prior_prodrate': prior_prodrate}

# Optimization settings
kwargs_opt = {'alpha_max': 0.25,    # step-size for x
              'alpha_cov': 0.001,   # step-size for cov
              'resample' : 1,       # number of allowed resamplings per iter
              'maxiter': 30,        # maximum number of 
              'alpha_maxiter': 5,
              'c1': 0.00001,
              'c2': 0.99}

# Simulation settings
kwargs_sim = {'npv_const': npv_params,
              'path_to_windpower': 'windpower_ensemble.npy',
              'ecalc_yamlfile'   : 'ecalc_config.yaml',
              'num_profiles'     : 50,
              'parallel'  : 10,
              'sim_limit' : 24000,
              'simoptions': [['mpi', 'mpirun -np 7'],
                             ['sim_flag', '--parsing-strictness=low']],
              'runfile'    : 'DROGON',
              'reportpoint': report_dates,
              'reporttype' : 'dates',
              'datatype'   : ['fopt', 'fgpt', 'fwpt', 'fwit', 'wthp a5', 'wthp a6']}
#---------------------------------------------------------------------------------------------

def delete_En_folders():
    ''' Deletes all folders starting with EN_'''
    for folder in glob('En_*'):                                             
        shutil.rmtree(folder)

def optimize_pareto_point(weight, save_folder):
    
    # Set save folder
    kwargs_opt.update({'save_folder': save_folder})

    # dummy objective
    def pareto(*args, save=False):
        intensity, npv, emissions = function(*args)
        f1 = emissions.sum(axis=1)
        f2 = npv

        ref = np.load('ref_data.npz', allow_pickle=True)
        f1_ref = ref['co2'].sum(axis=1).mean()
        f2_ref = ref['npv'].mean()

        f_tot = weight * f1/f1_ref + (1-weight) * (-f2/f2_ref)

        if save:
            np.savez(f'{save_folder}/result', 
                     pred_data = args[0],
                     co2 = emissions,
                     npv = npv,
                     Ico2 = intensity)
            
        return f_tot
    
    # Optimize 
    ensemble = Ensemble(kwargs_ens, flow(kwargs_sim), pareto)
    x0 = ensemble.get_state()
    corr = time_correlation(0.75, ensemble.state, 54)
    cov = corr2cov(corr, np.sqrt(np.diag(ensemble.get_cov())))
    print(cov)

    optimizer = LineSearch(fun=ensemble.function,
                           x=x0,
                           args=(cov,),
                           jac=ensemble.gradient,
                           hess=ensemble.hessian,
                           bounds=ensemble.get_bounds(),
                           **kwargs_opt)
    
    xf = optimizer.mean_state

    # Evaluate result
    dummy_func = lambda *args: pareto(*args, save=True)
    dummy_ensemble = Ensemble(kwargs_ens, flow(kwargs_sim), dummy_func)
    dummy_ensemble.function(xf)


if __name__ == '__main__':

    delete_En_folders()

    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        np.random.seed(7483643)
        optimize_pareto_point(weight=w, save_folder=f'pareto_front_wind/weight_{w}')