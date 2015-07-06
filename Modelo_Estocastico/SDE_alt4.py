#!/usr/bin/env python
"""
Version of the dengue 4 serotype model implemented for Stochpy
"""
import stochpy
import os
from itertools import cycle
import pylab as P
import numpy as np
from stochpy.modules import InterfaceCain


def plot(t, s, l):
    P.figure()
    s = np.vstack(s).T
    co = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    sy = cycle(['o', '^', '>', '<', 's', '*', '+', '1'])
    print(s.shape, len(t))
    for i in range(s.shape[1]):
        P.plot(t, s[:, i], co.next() + sy.next() + '-')
    P.legend(l, loc=0)

def agg_by_type(s, l):
    """
    return series aggregated by serotype
    :param s: series
    :param l: labels
    """
    s1 = np.zeros(s.shape[0])
    s2 = np.zeros(s.shape[0])
    s3 = np.zeros(s.shape[0])
    s4 = np.zeros(s.shape[0])
    for i, n in enumerate(l):
        if not n.startswith('I'):
            continue
        if n.endswith('1'):
            s1 += s[:, i]
        elif n.endswith('2'):
            s2 += s[:, i]
        elif n.endswith('3'):
            s3 += s[:, i]
        elif n.endswith('4'):
            s4 += s[:, i]


    return s1, s2, s3, s4


def plot_4_types(t, s, l, soma=True):
    """
    Plot all infectives for each serotype
    :param t: Times
    :param s: series
    :param l: labels
    """
    # print l
    P.figure()

    s1, s2, s3, s4 = agg_by_type(s, l)

    P.plot(t, s1, label=r'$I_{*1}$')

    P.plot(t, s2, label=r'$I_{*2}$')

    P.plot(t, s3, label=r'$I_{*3}$')

    P.plot(t, s4, label=r'$I_{*4}$')
    if soma:
        P.plot(t, s1+s2++s3+s4, label=r'$I_{*}$')
    P.legend(loc=0)
    P.xlabel('time')
    P.ylabel('individuals')
    P.savefig('4types.png', dpi=300)

def plot_xcorr(s1, s2):
    P.figure("Cross Correlation")
    P.xcorr(s1, s2, maxlags=50)


if __name__ == "__main__":
    # Loading the model
    smod = stochpy.SSA(IsInteractive=False)
    smod.model_dir = os.getcwd()
    smod.output_dir = os.getcwd()
    smod.Model('Dengue_full4.psc')


    # InterfaceCain.getCainInputfile(smod.SSA, 100, 100, 1)
    #    with open(os.path.join(stochpy.temp_dir,'cain_in.txt')) as f:
    #        with open('export.txt','w') as g:
    #            g.write(f.read())

    smod.DoStochSim(trajectories=1, mode='time', end=1000, method="TauLeaping", IsTrackPropensities=False)
    # smod.DoCainStochSim(endtime=1000, frames=10000, trajectories=1, solver="HomogeneousDirect2DSearch", IsTrackPropensities=False)
    smod.GetRegularGrid()
    smod.PlotSpeciesTimeSeries(species2plot=['I1', 'I2', 'I3', 'I4'])
    stochpy.plt.savefig(os.path.join(smod.output_dir, 'primary.png'), dpi=300)
    smod.PlotSpeciesTimeSeries(species2plot=['I12', 'I13', 'I14', 'I21', 'I23', 'I24', 'I31', 'I32', 'I34', 'I41', 'I42', 'I43'])
    stochpy.plt.savefig(os.path.join(smod.output_dir, 'secondary.png'), dpi=300)
    # smod.PlotSpeciesAutocorrelations(species2plot=['I4'], nlags=50)
    #smod.PlotAveragePropensitiesTimeSeries()
    #smod.PlotAverageSpeciesAutocorrelations()

    smod.PlotSpeciesDistributions(bin_size=50)

    t = smod.data_stochsim.time
    series = smod.data_stochsim.species
    sds = smod.data_stochsim_grid.species_standard_deviations
    l = smod.data_stochsim.species_labels
    s1, s2, s3, s4 = agg_by_type(series, l)
    plot_4_types(t, series, l)
    P.figure()
    P.specgram(s1, 256, t.max()/t.size)
    #plot_xcorr(s1, s2)
    #    plot(t,series,l)
    #    P.show()
    stochpy.plt.show()
