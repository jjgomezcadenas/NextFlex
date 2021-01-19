import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tics.histograms    import h1
from tics.histograms    import PlotLabels
from tics.histograms    import plot_histo
from  tics.system_of_units import *

from nextflex.reco_analysis import TrackRecoStats
from nextflex.reco_analysis import TrackRecoTiming
from nextflex.reco_analysis import GtrkStats



def plot_complexity(rbb, r1e, figsize=(14,7)):
    def plt_scatter(rb, label):
        plt.title (f"Reconstruction time versus complexity ({label})")
        plt.scatter(rb.tVoxelizationXYZ.xyz_bins, rb.tVoxelizationXYZ.time, marker='o')
        plt.xlabel(r"$x_{bins} \times y_{bins} \times z_{bins} / 10^6$")
        plt.ylabel("time (seconds)")

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 2, 1)
    plt_scatter(rbb, "bbonu")
    ax      = fig.add_subplot(1, 2, 2)
    plt_scatter(r1e, "1e")
    plt.tight_layout()
    plt.show()


def reco_gtrack_stats_histos(trs : TrackRecoStats,
                             n_evt_hits           = (0,1500),
                             energy_evt_hits      = (0,25),
                             tot_energy_evt_hits  = (0,2600),
                             n_voxels             = (0,100),
                             energy_voxels        = (0,500),
                             n_hits_voxels        = (0,200),
                             min_dist_voxels      = (0,10),
                             n_rec_gtrks          = (0,10),
                             figsize              = (14,10)):

    """
    Plots the following histograms

    1. n_evt_hits           : number of hits in the event (average)
    2. energy_evt_hits      : energy of hits
    3. tot_energy_evt_hits  : energy total of hits in keV
    4. n_voxels             : number of voxels (average)
    5. energy_voxels        : voxel energy (keV)
    6. n_hits_voxels        : average number of hits per voxel
    7. min_dist_voxels      : max of the minimum distances between voxels
    8. n_rec_gtrks          : number of reconstructed tracks

    """
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(4, 2, 1)
    n, b, mu, std    = h1(trs.n_evt_hits, bins=10, range=n_evt_hits,
                          stats = True)
    pltl = PlotLabels(x='Number of hits in event', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 2)
    energyKeV = np.array(trs.energy_evt_hits)
    n, b, mu, std    = h1(energyKeV, bins=10, range=energy_evt_hits,
                          stats = True)
    pltl = PlotLabels(x='Energy of Hits (keV)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 3)
    energyKeV = np.array(trs.tot_energy_evt_hits) / keV
    n, b, mu, std    = h1(energyKeV, bins=10, range=tot_energy_evt_hits,
                          stats = True)
    pltl = PlotLabels(x='Total energy of hits (keV)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 4)
    n, b, mu, std    = h1(trs.n_voxels, bins=10, range=n_voxels,
                          stats = True)
    pltl = PlotLabels(x='number of voxels', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 5)
    n, b, mu, std    = h1(trs.energy_voxels, bins=10, range=energy_voxels,
                          stats = True)
    pltl = PlotLabels(x='Voxel energy (keV)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 6)
    n, b, mu, std    = h1(trs.n_hits_voxels, bins=10, range=n_hits_voxels,
                          stats = True)
    pltl = PlotLabels(x='Hits per voxel', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 7)
    n, b, mu, std    = h1(trs.min_dist_voxels, bins=10,
                          range=min_dist_voxels,
                          stats = True)
    pltl = PlotLabels(x='minimum distance between voxels', y='events',
                          title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 8)
    n, b, mu, std    = h1(trs.n_rec_gtrks, bins=10, range=n_rec_gtrks,
                          stats = True)
    pltl = PlotLabels(x='Number of reconstructed tracks', y='events',
                         title=None)
    plot_histo(pltl, ax, legend=True)
    plt.tight_layout()


def reco_gtrack_timing_histos(trt : TrackRecoTiming,
                              t_evt_hits       = (0,200),
                              t_true_extrema   = (0,200),
                              t_voxelize_hits  = (0,200),
                              t_graph_tracks   = (0,200),
                              figsize=(14,10)):
    """
    Plots the following histograms

    1. t_evt_hits        : time to load event hits
    2. t_true_extrema    : time to compute true extrema
    3. t_voxelize_hits   : time to voxelize hits
    4. t_graph_tracks    : time to compute graph tracks

    """

    fig = plt.figure(figsize=figsize)

    ax      = fig.add_subplot(4, 2, 1)
    tms = np.array(trt.t_evt_hits) * 1000
    n, b, mu, std    = h1(tms, bins=10, range=t_evt_hits, stats = True)
    pltl = PlotLabels(x='Time to load Event Hits (ms)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 2)
    tms = np.array(trt.t_true_extrema) * 1000
    n, b, mu, std    = h1(tms, bins=10, range=t_true_extrema, stats = True)
    pltl = PlotLabels(x='Time to compute True extrema  (ms)',
                      y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 3)
    tms = np.array(trt.t_voxelize_hits) * 1000
    n, b, mu, std    = h1(tms, bins=10, range=t_voxelize_hits, stats = True)
    pltl = PlotLabels(x='Time to voxelize hits', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 4)
    tms = np.array(trt.t_graph_tracks) * 1000
    n, b, mu, std    = h1(tms, bins=10, range=t_graph_tracks, stats = True)
    pltl = PlotLabels(x='Time to create GTracks', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    plt.tight_layout()


def reco_gtrack_blobs_histos(gts : GtrkStats, figsize=(14,10)):

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(3, 2, 1)
    n, b, mu, std    = h1(gts.NumberOfVoxels, bins=10, range=[0,40], stats = True)
    pltl = PlotLabels(x='Number of voxels in gtrack', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(3, 2, 2)
    n, b, mu, std    = h1(gts.TrackLength, bins=10, range=[0,250], stats = True)
    pltl = PlotLabels(x='Track Length', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(3, 2, 3)
    n, b, mu, std    = h1(gts.EnergyBlob1, bins=10, range=[0,2000], stats = True)
    pltl = PlotLabels(x='Energy blob1', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(3, 2, 4)
    n, b, mu, std    = h1(gts.EnergyBlob2, bins=10, range=[0,2000], stats = True)
    pltl = PlotLabels(x='Energy blob2', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(3, 2, 5)

    n, b, mu, std    = h1(gts.NumberVoxelsBlob1, bins=10, range=[0,20], stats = True)
    pltl = PlotLabels(x='Number of voxels blob1', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(3, 2, 6)
    n, b, mu, std    = h1(gts.NumberVoxelsBlob2, bins=10, range=[0,20], stats = True)
    pltl = PlotLabels(x='Number of voxels blob2', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    plt.tight_layout()
