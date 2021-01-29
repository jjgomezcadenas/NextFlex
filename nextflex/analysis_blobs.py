import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from invisible_cities.core.core_functions import in_range
from  tics.system_of_units import *
from  nextflex.reco_analysis import select_gtrack_topology
from dataclasses import dataclass


@dataclass
class BAD:
    """
    Blob Analysis Description (BAD)

    st  : single track
    eff : efficiency
    bb  : double beta
    1e  : single track electrons
    rf  : rejection factor
    fm  : figure of merit

    """
    total_events_bb : float
    total_events_1e : float
    st_eff_bb       : float = 0
    st_eff_1e       : float = 0
    energy_eff_bb   : float = 0
    energy_eff_1e   : float = 0
    blobs_eff_bb    : float = 0
    blobs_eff_1e    : float = 0
    bb_1e_rf        : float = 0
    bb_1e_fm        : float = 0

    def __repr__(self):
        s = f"""
        single track eff 1e          = {self.st_eff_1e:5.2f}
        energy       eff bb          = {self.energy_eff_bb:5.2f}
        energy       eff 1e          = {self.energy_eff_1e:5.2f}
        blobs        eff bb          = {self.blobs_eff_bb:5.2f}
        blobs        eff 1e          = {self.blobs_eff_1e:5.3f}
        blobs suppresion factor      = {self.bb_1e_rf:5.2f}
        blobs figure of merit        = {self.bb_1e_fm:5.2f}
        """
        return s

    __str__ = __repr__


def select_blobs(rbb, eb_cut):
    """
    returns a gtrack summary dataframe with cuts on
    the energy of the blobs.

    """
    eb1 = rbb[rbb.energy_b1.values > eb_cut]
    return eb1[eb1.energy_b2.values > eb_cut]


def select_distance_cut(ddf, gtbb, dcut):
    """
    Select events in gtrack dataframe gtbb
    which verify that the distance between
    the reconstructed and the true extrema
    (passed by dataframe ddf) is smaller than
    dcut

    """
    def select_d1d2(ddf, dcut):
        eb1 = ddf[ddf.d1 < dcut]
        return eb1[eb1.d2 < dcut]

    ddf2 = select_d1d2(ddf, dcut)
    events_ddf = ddf2.event_id.values
    events_gtbb = gtbb.event_id.values
    return events_ddf


def select_topology_and_energy(gtbb, gt1e, energy_range=(2400, 2500)):
    """
    Selects events with single tracks (topology selection) and
    in the energy range 100 keV around Qbb (energy selection)

        - gtbb : a gtrack summary data frame or bb0nu events
        - gt1e : a gtrack summary data frame or 1e events

    """
    def select(tbb):
        t1rbb = select_gtrack_topology(tbb, topology = "single")
        return t1rbb[in_range(t1rbb.energy, *energy_range)]

    return select(gtbb), select(gt1e)


def distance_reco_true(sgt, ste, event_range=(0,-1)):
    """
    Takes a sorted gtrack dataframe (sgt) and a sorted
    true extrema dataframe (ste), and computes a dataframe
    with the distances between the true extrema and the
    extrema found by the reconstruction in the gtrack

    """
    def select_gt(sgt, evt):
        return sgt.loc[slice(evt, evt), :]

    def select_te(ste, gtrk):
        return ste[ste.evt_number == gtrk.event_id.values[0]]

    def select_te_xyz(tre):
        e1 = tre[tre.extreme_number==0]
        e2 = tre[tre.extreme_number==1]
        pe1  = e1[['x','y','z']].values[0]
        pe2  = e2[['x','y','z']].values[0]
        return pe1, pe2

    def select_te_energy(tre):
        e1 = tre[tre.extreme_number==0]
        e2 = tre[tre.extreme_number==1]
        return e1.energy.values[0], e2.energy.values[0]

    def select_gt_xyz(gtrk):
        pg1 = gtrk[['x_e1','y_e1', 'z_e1']].values[0]
        pg2 = gtrk[['x_e2','y_e2', 'z_e2']].values[0]
        return pg1, pg2

    def select_gt_energy(gtrk):
        return gtrk.energy_e1.values[0], gtrk.energy_e2.values[0]

    def sort_te_by_distance_to_gt(pe, pg):
        d1 = np.linalg.norm(pe[0] - pg[0])
        d2 = np.linalg.norm(pe[0] - pg[1])
        if d1 < d2:
            e1 = pe[0]
            e2 = pe[1]
        else:
            e1 = pe[1]
            e2 = pe[0]
        return e1, e2

    def distance_te_gt(pe, pg):
        d1 = np.linalg.norm(pe[0] - pg[0])
        d2 = np.linalg.norm(pe[1] - pg[1])
        return d1, d2

    evt_i = event_range[0]
    evt_f = event_range[1]
    D1 = []
    D2 = []
    EVT = []
    for i, evt in enumerate(sgt.index[evt_i:evt_f]):

        gtrk = select_gt(sgt, evt)
        tre  = select_te(ste, gtrk)

        #print(f" \n event number = {i}, event_id = {gtrk.event_id.values[0]}")

        pe = select_te_xyz(tre)
        pg = select_gt_xyz(gtrk)

        #print(f" pg1 ={pg[0]}")
        #print(f" pg2 ={pg[1]}")

        pe = sort_te_by_distance_to_gt(pe, pg)

        #print(f" pe1 ={pe[0]}")
        #print(f" pe2 ={pe[1]}")

        d1, d2 = distance_te_gt(pe, pg)

        #print(f" d1 ={d1},  d2 = {d2}")

        D1.append(d1)
        D2.append(d2)
        EVT.append(gtrk.event_id.values[0])

    data = {'event_id' : EVT,'d1' : D1, 'd2' : D2}
    return pd.DataFrame(data).sort_values(by=['event_id'])



def select_blobs_eff_energy(rbb, gtbb, gt1e, eb_range = (150,300),
                            figsize=(14,7)):
    """
    Computes curves of efficiency and figures of merit for
    signal and background.

    """
    def select(rbb, ebcut):
        eb1 = rbb[rbb.energy_b1.values > ebcut]
        return eb1[eb1.energy_b2.values > ebcut]

    eBB = []
    e1E = []
    total_events = rbb.trackRecoEventStats.e_total
    eb_cut = np.linspace(*eb_range,10)

    eBB = np.array([len(select_blobs(gtbb, ecut)) / len(gtbb)\
                    for ecut in eb_cut])
    e1E = np.array([len(select_blobs(gt1e, ecut)) / len(gt1e) \
                    for ecut in eb_cut])

    fm1 = eBB / e1E
    fm2 = eBB / np.sqrt(e1E)

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(2, 2, 1)
    plt.plot(eb_cut, eBB)
    plt.xlabel("Eb cut (keV)")
    plt.ylabel("bb0nu efficiency (%)")
    ax      = fig.add_subplot(2, 2, 2)
    plt.plot(eb_cut, e1E)
    plt.xlabel("Eb cut (keV)")
    plt.ylabel("1e efficiency (%)")
    ax      = fig.add_subplot(2, 2, 3)
    plt.plot(eb_cut, fm1)
    plt.xlabel("Eb cut (keV)")
    plt.ylabel("S/N ")
    ax      = fig.add_subplot(2, 2, 4)
    plt.plot(eb_cut, fm2)
    plt.xlabel("Eb cut (keV)")
    plt.ylabel("S/Sqrt(N) ")
    plt.tight_layout()
    plt.show()



def select_two_blobs(fbb, eb_cut):
    """
    Computes the selection efficiency for two blobs with
    energy above eb_cut

    """

    def select_blobs(rbb, eb_cut):
        eb1 = rbb[rbb.energy_b1.values > eb_cut]
        return eb1[eb1.energy_b2.values > eb_cut]

    brbb  = select_blobs(fbb, eb_cut)
    return brbb, len(brbb) / len(fbb)


def select_energy(gtbb, energy_range=(2400, 2500)):
    """
    Selects events in the energy range 100 keV around Qbb

    """
    def energy_sel(tbb):
        return tbb[in_range(tbb.energy, *energy_range)]


    gebb = energy_sel(gtbb)

    return gebb, len(gebb) / len(gtbb)


def selection_efficiency(rbb, gtbb, energy_range=(2400, 2500),
                         eb_cut=200):
    """
    Computes the selection efficiency for a selection
    """
    def output_eff():
        total_events = rbb.trackRecoEventStats.e_total
        print("Selection efficiency:")
        print(f"- single track topology  = {len(t1rbb) / total_events}")
        print(f"- energy cut             = {len(erbb) / len(t1rbb)}")
        print(f"- 2-blob cut             = {len(brbb) / len(erbb)}")
        print(f"- total efficiency       = {len(brbb) / total_events}")

    t1rbb = select_gtrack_topology(gtbb, topology = "single")
    erbb  = t1rbb[in_range(t1rbb.energy, *energy_range)]
    brbb  = select_blobs(erbb, eb_cut)
    output_eff()


def d12_eff(ddf, d_range = (0,50), figsize=(14,7)):
    """
    Computes curves of efficiency as a function of d_cut

    """
    def select(ddf, dcut):
        eb1 = ddf[ddf.d1 < dcut]
        return eb1[eb1.d2 < dcut]

    eBB = []
    e1E = []
    d_cut = np.linspace(*d_range,10)

    eDC = np.array([len(select(ddf, dcut)) / len(ddf) for dcut in d_cut])
    return d_cut, eDC
    # fig = plt.figure(figsize=figsize)
    # ax      = fig.add_subplot(1, 1, 1)
    # plt.plot(d_cut, eDC)
    # plt.xlabel("distance cut (mm)")
    # plt.ylabel("efficiency")
    # plt.tight_layout()
    # plt.show()


def blob_energy(ebb, e1e, eb_cut=200, figsize=(14,7)):
    """
    Plots the energy of blob1 vs blob2 for bb0nu events and
    1e events:
    - ebb : a gtrack summary data frame or bb0nu events
    - e1e : a gtrack summary data frame or 1e events
    - eb_cut: values of the cuts (to be plotted as v and h lines)
    """
    def plt_scatter(rb,label):

        plt.axvline(x = eb_cut)
        plt.axhline(y = eb_cut)
        plt.title (f"Eb1 vs Eb2 ({label})")
        plt.scatter(rb.energy_b1 , rb.energy_b2 , marker='o')

        plt.xlabel("Eb1 (keV)")
        plt.ylabel("Eb2 (keV)")

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 2, 1)
    plt_scatter(ebb, "bbonu")
    ax      = fig.add_subplot(1, 2, 2)
    plt_scatter(e1e, "1e")
    plt.tight_layout()
    plt.show()


def energy_resolution(energy, xlim, bins=25, range=(2400,2460), figsize=(10,10),
                      alpha=0.6):
    """
    Plots the event energy and fits a gaussian
    - energy : energy vector (in keV)
    - xlim   : limits

    """

    fig = plt.figure(figsize=figsize)

    xmin = xlim[0]
    xmax = xlim[1]

    ax      = fig.add_subplot(1, 1, 1)
    mu, std = norm.fit(energy)
    x = np.linspace(xmin, xmax, 25)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)

    plt.hist(energy, bins=bins, density=True, range=range,
             alpha=alpha, label=f'\u03C3 = {std:5.2f} keV')
    plt.xlabel(f'energy (keV)')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    return mu, std
