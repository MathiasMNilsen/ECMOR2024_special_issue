import matplotlib as plt
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
import mplcyberpunk
import seaborn as sns
import os

mpl.rcParams.update({'axes.grid'        : True,
                     'grid.linestyle'   : '-',
                     'grid.color'       : 'gray',
                     'grid.alpha'       : 0.4,
                     'axes.linewidth'   : 0.5,
                     'legend.frameon'   : False,
                     'xtick.major.size' : 0,
                     'ytick.major.size' : 0,
                     'xtick.minor.size' : 0,
                     'ytick.minor.size' : 0,})

def get_rates(pred_file, dates, ne=50):

    pred_data = pred_file['pred_data']
    ni = len(pred_data) - 1
    rates = {'FOPR': np.zeros((ne,ni)), 'FGPR': np.zeros((ne,ni)), 'FWPR': np.zeros((ne,ni)), 'CO2R': np.zeros((ne,ni))}
    get_data  = lambda i, key: pred_data[i+1][key].squeeze() - pred_data[i][key].squeeze()
    
    for d in range(ni):
        
        days = (dates[d+1] - dates[d]).days
        rates['FOPR'][:,d] = get_data(d, 'fopt')/days
        rates['FGPR'][:,d] = get_data(d, 'fgpt')/days
        rates['FWPR'][:,d] = get_data(d, 'fwpt')/days
        rates['CO2R'][:,d] = pred_file['co2'][:,d]

    return rates

def plot_ref():
    
    file = np.load('ref_data.npz', allow_pickle=True)
    rep_dates = pd.date_range('2020-07-01', '2025-01-01', freq='MS').to_pydatetime()

    print('CO2:', np.mean(file['co2'].sum(1))/1000)
    print('NPV:', np.mean(file['npv'])/1e9)
    
    # Get rates
    rates = get_rates(file, rep_dates)

    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10,5))
    fig.autofmt_xdate(rotation=20, ha='center')
    p = [(0,0), (0,1), (1,0), (1,1)]
    titles = [r'FOPR [10$^3$ Sm$^3$/day]', r'FGPR [10$^3$ Sm$^3$/day]', r'FWPR [10$^3$ Sm$^3$/day]', r'Emissions rate [10$^3$ tonnes/day]']

    for k, key in enumerate(rates.keys()):
        rates[key] = np.insert(rates[key], 0, rates[key][:,0], axis=1)
        rates[key] = np.insert(rates[key], -1, rates[key][:,-1], axis=1)
        rates[key] = rates[key]/1000

        ax[p[k]].fill_between(rep_dates, np.max(rates[key], axis=0), np.min(rates[key], axis=0), alpha=0.6, color='dimgray')
        ax[p[k]].plot(rep_dates, rates[key].mean(axis=0), color='lavender', lw=1.25)
        ax[p[k]].set_xlim(rep_dates[0], rep_dates[-1])
        ax[p[k]].set_title(titles[k])
        ax[p[k]].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        ax[p[k]].tick_params(axis='x', labelsize=8)


    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    plt.show()
    fig.savefig('ref_wind', dpi=300)


def compare_pred_data(file1, file2, label1='', label2=''):

    rep_dates = pd.date_range('2020-07-01', '2025-01-01', freq='MS').to_pydatetime()

    print('CO2:', np.mean(file1['co2'].sum(1))/1000)
    print('NPV:', np.mean(file1['npv'])/1e9)
    
    # Get rates
    rates1 = get_rates(file1, rep_dates)
    rates2 = get_rates(file2, rep_dates)

    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10,5))
    fig.autofmt_xdate(rotation=20, ha='center')
    p = [(0,0), (0,1), (1,0), (1,1)]
    titles = [r'FOPR [10$^3$ Sm$^3$/day]', r'FGPR [10$^3$ Sm$^3$/day]', r'FWPR [10$^3$ Sm$^3$/day]', r'Emissions rate [10$^3$ tonnes/day]']

    for k, key in enumerate(rates1.keys()):

        rates1[key] = np.insert(rates1[key], 0, rates1[key][:,0], axis=1)
        rates1[key] = np.insert(rates1[key], -1, rates1[key][:,-1], axis=1)
        rates1[key] = rates1[key]/1000

        ax[p[k]].fill_between(rep_dates, np.max(rates1[key], axis=0), np.min(rates1[key], axis=0), alpha=0.5, color='teal')
        ax[p[k]].plot(rep_dates, rates1[key].mean(axis=0), color='teal', lw=1.0, label=label1)
        ax[p[k]].set_xlim(rep_dates[0], rep_dates[-1])
        ax[p[k]].set_title(titles[k])
        ax[p[k]].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        ax[p[k]].tick_params(axis='x', labelsize=8)

        rates2[key] = np.insert(rates2[key], 0, rates2[key][:,0], axis=1)
        rates2[key] = np.insert(rates2[key], -1, rates2[key][:,-1], axis=1)
        rates2[key] = rates2[key]/1000

        ax[p[k]].fill_between(rep_dates, np.max(rates2[key], axis=0), np.min(rates2[key], axis=0), alpha=0.5, color='dimgray')
        ax[p[k]].plot(rep_dates, rates2[key].mean(axis=0), color='dimgray', lw=1, label=label2)

    ax[p[0]].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15)
    plt.show()
    #fig.savefig('compare', dpi=300)


if __name__ == '__main__':

    #plot_ref()

    f1 = np.load('pareto_front_wind/weight_1.0/result.npz', allow_pickle=True)
    f2 = np.load('pareto_front_nowind/weight_1.0/result.npz', allow_pickle=True)
    #f2 = np.load('ref_data_nowind.npz', allow_pickle=True)
    l1 = r'Wind and gas power: $\omega=1.0$'
    l2 = r'Only gas power: $\omega=1.0$'
    compare_pred_data(f1, f2, l1, l2)


    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        file_wind = np.load(f'pareto_front_wind/weight_{w}/result.npz', allow_pickle=True)

        npv_mean = np.mean(file_wind['npv']/1e9)
        npv_std  = np.std(file_wind['npv']/1e9)

        co2_mean = np.mean(file_wind['co2'].sum(1)/1000)
        co2_std  = np.std(file_wind['co2'].sum(1)/1000)

        I_mean = np.mean(file_wind['Ico2'])
        I_std  = np.std(file_wind['Ico2'])


        print('\n')
        print(f'Wind: w = ', w)
        print(f'NPV: {round(npv_mean,2)} (std = {round(npv_std, 2)}) USD')
        print(f'CO2: {round(co2_mean,2)} (std = {round(co2_std, 2)}) kilo tonnes')
        print(f'I: {round(I_mean,2)} (std = {round(I_std, 2)}) kg/toe')