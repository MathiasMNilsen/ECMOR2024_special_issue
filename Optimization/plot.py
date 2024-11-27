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

def get_optimize_results(folder, filekey):
    '''
    Retrieve the debug values from a folder containing debug analysis files.

    Parameters
    ----------
        folder (str) : The path to the folder containing debug analysis files.
        filekey (str) : The key to extract from each debug analysis file.

    Returns
    -------
        numpy.ndarray: An array containing the values associated with the specified 'filekey'
        from each debug analysis file found in the folder.

    Example use
    -----------
    >>> obj_values = get_debug_values('/path/to/results', 'obj_func_values')
    '''
    files_in_dir = os.listdir(folder)
    debug_files  = [name for name in files_in_dir if 'optimize_result' in name]
    iterations   = len(debug_files)

    values = []
    for iter in range(iterations):
        info = np.load(folder + f'/optimize_result_{iter}.npz')

        try:
            values.append(info[filekey])
        except:
            raise ValueError(f'{filekey} not in npz. The possible keys are: {info.files}')
    
    return np.asarray(values)


def get_pred_data(file):

    pred_data = np.load(file, allow_pickle=True)['pred_data']
    
    fopr = []
    fgpr = []
    fwpr = []

    for d in range(len(pred_data)):

        fopr.append(pred_data[d]['fopt'].squeeze())
        fgpr.append(pred_data[d]['fgpt'].squeeze())
        fwpr.append(pred_data[d]['fwpt'].squeeze())

    fopr = np.array(fopr).T
    fgpr = np.array(fgpr).T
    fwpr = np.array(fwpr).T

    return fopr, fgpr, fwpr

def plot_pareto():

    weights =  [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(figsize=(9,7.5))
    colors = ['tab:blue', 'tab:red', 'teal', 'tab:brown', 'black']
    
    npv_w = []
    npv_g = []
    co2_w = []
    co2_g = []

    # inset Axes
    axinsw = ax.inset_axes([0.0, 1.1, 1.0, 0.5],
                           xlim=(180, 240), 
                           ylim=(24, 29.5))
    ax.indicate_inset_zoom(axinsw, edgecolor='tab:blue')
    axinsw.spines['bottom'].set_color('tab:blue')
    axinsw.spines['top'].set_color('tab:blue') 
    axinsw.spines['right'].set_color('tab:blue')
    axinsw.spines['left'].set_color('tab:blue')
    axinsw.tick_params(axis='x', which='major', pad=7.5)
    axinsw.set_title('Wind and Gas power')


    axinsg = ax.inset_axes([1.1, 0.0, 0.5, 1.0],
                           xlim=(300, 347), 
                           ylim=(23, 29))
    ax.indicate_inset_zoom(axinsg, edgecolor='tab:red')
    axinsg.spines['bottom'].set_color('tab:red')
    axinsg.spines['top'].set_color('tab:red') 
    axinsg.spines['right'].set_color('tab:red')
    axinsg.spines['left'].set_color('tab:red')
    axinsg.tick_params(axis='y', which='major', pad=7.5)
    axinsg.set_title('Only Gas power')

    for w, weight in enumerate(weights):

        # Wind
        res = np.load(f'pareto_front_wind/weight_{weight}/result.npz')
        npv = res['npv']/1e9
        co2 = res['co2'].sum(axis=1)/1e3

        npv_w.append(npv.mean())
        co2_w.append(co2.mean())
    
        ax.scatter(co2, npv, color=colors[w], alpha=0.3, zorder=1, s=3)
        ax.scatter(co2.mean(), npv.mean(), color=colors[w], s=60, zorder=2, label=rf'$\omega={weight}$')

        axinsw.scatter(co2, npv, color=colors[w], alpha=0.6, zorder=1, s=5)
        axinsw.scatter(co2.mean(), npv.mean(), color=colors[w], s=60, zorder=2)



        # Gas
        res = np.load(f'pareto_front_nowind/weight_{weight}/result.npz')
        npv = res['npv']/1e9
        co2 = res['co2'].sum(axis=1)/1e3

        npv_g.append(npv.mean())
        co2_g.append(co2.mean())

        ax.scatter(co2, npv, color=colors[w], alpha=0.3, zorder=1, s=3)
        ax.scatter(co2.mean(), npv.mean(), color=colors[w], s=60, zorder=2)

        axinsg.scatter(co2, npv, color=colors[w], alpha=0.6, zorder=1, s=5)
        axinsg.scatter(co2.mean(), npv.mean(), color=colors[w], s=60, zorder=2)

    # ploly fit
    pw = np.poly1d(np.polyfit(co2_w, npv_w, 3))
    xw = np.linspace(co2_w[-1], co2_w[0], 50)
    ax.plot(xw, pw(xw), color='dimgray', zorder=0, ls='dashed', alpha=0.7, label=r'3$^{rd} deg. polyfit$')
    axinsw.plot(xw, pw(xw), color='dimgray', zorder=0, ls='dashed', alpha=0.7)

    pg = np.poly1d(np.polyfit(co2_g, npv_g, 3))
    xg = np.linspace(co2_g[-1], co2_g[0], 50)
    ax.plot(xg, pg(xg), color='dimgray', zorder=0, ls='dashed', alpha=0.7)
    axinsg.plot(xg, pg(xg), color='dimgray', zorder=0, ls='dashed', alpha=0.7)


    ax.set_xlim(100, 350)
    ax.set_ylim(-1, 30)

    ax.set_ylabel(r'NPV [Billion $]')
    ax.set_xlabel(r'CO$_2$ emissions [kilo tonnes]')

    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.62, right=0.64)
    plt.draw()
    fig.savefig('pareto', dpi=300)


def plot_conv():

    weights =  [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(ncols=len(weights), figsize=(14,3), sharex=True)
    colors  = ['tab:blue', 'tab:red', 'tab:green', 'tab:brown', 'black', 'pink']

    for w, we in enumerate(weights):
        
        # Wind
        res = f'pareto_front_wind/weight_{we}'    
        fun = get_optimize_results(res, 'fun')
        ax[w].plot(fun, label=rf'$\omega={we}$', color=colors[w])

        # Gas
        res = f'pareto_front_nowind/weight_{we}'    
        fun = get_optimize_results(res, 'fun')
        ax[w].plot(fun, ls='dashed', color=colors[w])


        ax[w].legend()

    plt.tight_layout()
    plt.draw()
    #fig.savefig('conv')

def plot_controls():

    nt = 54
    weights =  [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(nrows=len(weights), ncols=3, sharex=True, sharey='col', figsize=(12,8))
    fig.autofmt_xdate(ha='center')

    colors = ['tab:blue', 'tab:red', 'teal', 'tab:brown', 'black', 'pink']
    #colors = ['steelblue', 'indianred', 'teal', 'tab:brown']
    colgas = ['darkblue', 'darkred', 'darkgreen', 'brown', 'black', 'pink']

    t = pd.date_range(start='2020-07-01', end='2025-01-01', freq='MS').to_pydatetime()

    for w, we in enumerate(weights):

        # Wind
        x = get_optimize_results(f'pareto_front_wind/weight_{we}', 'x')[-1]
        qA5, qA6 = np.reshape(x[:2*nt]*8000, (nt,2)).T
        qOP = x[2*nt:]*18000

        qA5 = np.insert(qA5, 0, qA5[0])
        qA6 = np.insert(qA6, 0, qA6[0])
        qOP = np.insert(qOP, 0, qOP[0])
        #t = np.arange(len(qA5))

        ax[w,0].step(t, qA5, alpha=0.8, color=colors[w], zorder=1, lw=1.75)
        ax[w,1].step(t, qA6, alpha=0.8, color=colors[w], zorder=1, lw=1.75)
        ax[w,2].step(t, qOP, alpha=0.8, color=colors[w], zorder=1, lw=1.75, label=rf'$\omega={we}$')

        ax[w,2].legend(loc='upper right')

        #mplcyberpunk.add_gradient_fill(ax[w,0], 0.3)
        #mplcyberpunk.add_gradient_fill(ax[w,1], 0.3)
        #mplcyberpunk.add_gradient_fill(ax[w,2], 0.3)

        # Gas
        x = get_optimize_results(f'pareto_front_nowind/weight_{we}', 'x')[-1]
        qA5, qA6 = np.reshape(x[:2*nt]*8000, (nt,2)).T
        qOP = x[2*nt:]*18000
        qA5 = np.insert(qA5, 0, qA5[0])
        qA6 = np.insert(qA6, 0, qA6[0])
        qOP = np.insert(qOP, 0, qOP[0])
        ax[w,0].step(t, qA5, alpha=0.6, color='gray', lw=1, zorder=0)
        ax[w,1].step(t, qA6, alpha=0.6, color='gray', lw=1, zorder=0)
        ax[w,2].step(t, qOP, alpha=0.6, color='gray', lw=1, zorder=0)

        ax[w,0].tick_params(axis='x', labelsize=8)
        ax[w,1].tick_params(axis='x', labelsize=8)
        ax[w,2].tick_params(axis='x', labelsize=8)

        ax[w,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        ax[w,1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        ax[w,2].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        

    ax[0,0].set_xlim(t[0], t[-1])

    ax[0,0].set_ylim(-600, 8300)
    ax[0,1].set_ylim(-600, 8300)
    ax[0,2].set_ylim(-1000, 18500)
    
    ax[0,0].set_title(r'WWIR A5 [Sm$^3$/day]')
    ax[0,1].set_title(r'WWIR A6 [Sm$^3$/day]')
    ax[0,2].set_title(r'target FOPR [Sm$^3$/day]')

    ax[0,0].tick_params(axis='x', labelsize=8)
    ax[0,1].tick_params(axis='x', labelsize=8)
    ax[0,2].tick_params(axis='x', labelsize=8)
    
    #fig.legend() 
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.18)
    plt.draw()
    fig.savefig('controls', dpi=300)


def plot_intensity():

    weights =  [0.0, 0.25, 0.5, 0.75]
    fig, ax = plt.subplots()
    colors = ['tab:blue', 'tab:red', 'teal', 'tab:brown', 'black']

    for w, weight in enumerate(weights):

        # Wind
        res = np.load(f'pareto_front_wind/weight_{weight}/result.npz')
        npv = res['npv']/1e9
        Ico2 = res['Ico2']
        #ax.scatter(Ico2, npv, color=colors[w], alpha=0.3, zorder=1, s=3)
        ax.scatter(Ico2.mean(), npv.mean(), color=colors[w], s=60, zorder=2)
        ax.errorbar(Ico2.mean(), npv.mean(), npv.std(), Ico2.std(), color=colors[w])


        # Gas
        res = np.load(f'pareto_front_nowind/weight_{weight}/result.npz')
        npv = res['npv']/1e9
        Ico2 = res['Ico2']

        ax.scatter(Ico2.mean(), npv.mean(), color=colors[w], s=60, zorder=2)
        ax.errorbar(Ico2.mean(), npv.mean(), npv.std(), Ico2.std(), color=colors[w])

    ax.set_ylabel(r'NPV [Billion $]')
    ax.set_xlabel(r'CO$_2$ intenisty [kg/toe]')

    plt.legend()
    plt.tight_layout()
    plt.draw()



if __name__ == '__main__':

    plot_pareto()
    #plot_conv()
    plot_controls()
    #plot_intensity()


    plt.show()