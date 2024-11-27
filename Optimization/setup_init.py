import glob
import shutil
from glob import glob
import numpy as np
import pandas as pd

from popt.loop.ensemble import Ensemble
from simulator.opm import flow
from objective import function

init_injrate = [['mean', 108*[6000]],
                ['var', 64000],
                ['limits', 0, 6000]]

init_prodrate = [['mean', 54*[18000]],
                 ['var', 324000],
                 ['limits', 0, 18000]]

report_dates = pd.date_range('2020-08-01', '2025-01-01', freq='MS').to_pydatetime().tolist()

# NPV parameters
npv_params = [['wop', 600.0], # oil price [USD/Sm³]
              ['wgp', 17.0],  # gas price [USD/Sm³]
              ['wwp', 38.0],  # cost of water disposal  [USD/Sm³]
              ['wwi', 18.0],  # cost of water injection [USD/Sm³]
              ['wem', 0.15],  # co2 tax [USD/kg]
              ['disc', 0.08]] # discount factor

kwargs_ens = {'ne': 50,
              'state': ['injrate', 'prodrate'],
              'transform': True,
              'num_models': 50,
              'prior_injrate': init_injrate,
              'prior_prodrate': init_prodrate}

kwargs_sim = {'npv_const': npv_params,
              'path_to_windpower': './windpower_ensemble.npy',
              'ecalc_yamlfile': 'ecalc_config.yaml',
              'num_profiles': 50,
              'parallel': 10,
              'sim_limit': 24000,
              'simoptions': [['mpi', 'mpirun -np 5'],
                             ['sim_flag', '--parsing-strictness=low']],
              'runfile': 'DROGON',
              'reportpoint': report_dates,
              'reporttype': 'dates',
              'datatype': ['fopt', 'fgpt', 'fwpt', 'fwit', 'wthp a5', 'wthp a6']}


def delete_En_folders():
    for folder in glob('En_*'):                                             
        shutil.rmtree(folder)

def plot_fopr():
    import matplotlib.pyplot as plt

    report_dates = pd.date_range('2020-07-01', '2025-01-01', freq='MS').to_pydatetime().tolist()
    print(len(report_dates))
    pred_data = np.load('ref_data.npz', allow_pickle=True)['pred_data']
    get_data  = lambda i, key: pred_data[i+1][key].squeeze() - pred_data[i][key].squeeze()

    fopr = []
    for d in range(len(pred_data)-1):
        days = (report_dates[d+1] - report_dates[d]).days
        fopr.append(get_data(d, 'fopt')/days)
    fopr.append(fopr[-1])
    fopr = np.array(fopr).T
    print(fopr.shape)

    fig, ax = plt.subplots()

    for n in range(fopr.shape[0]):
        ax.plot(fopr[n], color='tab:blue', alpha=0.5)

    ax.plot(fopr.mean(axis=0), color='black')

    #plt.show()
    fig.savefig('fopr')

    # save fopr
    np.savez('init_fopr', fopr.mean(axis=0))

if __name__ == '__main__':

    delete_En_folders()

    # set up no wind ensemble 
    #no_wind = np.zeros_like(np.load('./windpower_ensemble.npy'))
    #np.save('./zero_wind_ens', no_wind)

    def get_pred_data(*args):
        intensity, npv, emissions = function(*args)
        pred_data = args[0]
        np.savez('ref_data', 
                 pred_data=pred_data,
                 co2 = emissions,
                 npv = npv,
                 Ico2 = intensity)
        return 0
    
    ensemble = Ensemble(kwargs_ens, flow(kwargs_sim), get_pred_data)
    ensemble.function(ensemble.get_state())

    plot_fopr()
