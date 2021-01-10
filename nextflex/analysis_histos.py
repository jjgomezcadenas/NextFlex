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


def reco_gtrack_stats_histos(trs : TrackRecoStats,
                             rNumberMCHits      = (0,1500),
                             rEnergyMCHits      = (0,25),
                             rTotalEnergyMCHits = (0,2600),
                             rNumberOfVoxels    = (0,100),
                             rVoxelEnergyKeV    = (0,500),
                             rHitsPerVoxel      = (0,200),
                             rMinimumDistVoxels = (0,10),
                             rNumberRecTrks     = (0,10),
                             figsize            = (14,10)):
    """
    Plots the following histograms

    1. NumberMCHits
    2. EnergyMCHits
    3. TotalEnergyMCHits
    4. NumberOfVoxels
    5. VoxelEnergyKeV
    6. HitsPerVoxel
    7. MinimumDistVoxels
    8. NumberRecTrks

    """
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(4, 2, 1)
    n, b, mu, std    = h1(trs.NumberMCHits, bins=10, range=rNumberMCHits,
                          stats = True)
    pltl = PlotLabels(x='Number of MC Hits', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 2)
    energyKeV = np.array(trs.EnergyMCHits) / keV
    n, b, mu, std    = h1(energyKeV, bins=10, range=rEnergyMCHits,
                          stats = True)
    pltl = PlotLabels(x='Energy Hits (keV)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 3)
    energyKeV = np.array(trs.TotalEnergyMCHits) / keV
    n, b, mu, std    = h1(energyKeV, bins=10, range=rTotalEnergyMCHits,
                          stats = True)
    pltl = PlotLabels(x='Total energy (keV)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 4)
    n, b, mu, std    = h1(trs.NumberOfVoxels, bins=10, range=rNumberOfVoxels,
                          stats = True)
    pltl = PlotLabels(x='number of voxels', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 5)
    n, b, mu, std    = h1(trs.VoxelEnergyKeV, bins=10, range=rVoxelEnergyKeV,
                          stats = True)
    pltl = PlotLabels(x='Voxel energy (keV)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 6)
    n, b, mu, std    = h1(trs.HitsPerVoxel, bins=10, range=rHitsPerVoxel,
                          stats = True)
    pltl = PlotLabels(x='Hits per voxel', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 7)
    n, b, mu, std    = h1(trs.MinimumDistVoxels, bins=10,
                          range=rMinimumDistVoxels,
                          stats = True)
    pltl = PlotLabels(x='minimum distance between voxels', y='events',
                          title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 8)
    n, b, mu, std    = h1(trs.NumberRecTrks, bins=10, range=rNumberRecTrks,
                          stats = True)
    pltl = PlotLabels(x='Number of reconstructed tracks', y='events',
                         title=None)
    plot_histo(pltl, ax, legend=True)
    plt.tight_layout()


def reco_gtrack_timing_histos(trt : TrackRecoTiming, figsize=(14,10)):
    """
    Plots the following histograms

    1. TimeMcHits
    2. TimeEvtHits
    3. TimeVoxHits
    4. TimeGT

    """
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(4, 2, 1)
    tms = np.array(trt.TimeMcHits) * 1000
    n, b, mu, std    = h1(tms, bins=10, range=[0,20], stats = True)
    pltl = PlotLabels(x='Time to load Hits (ms)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 2)
    tms = np.array(trt.TimeEvtHits) * 1000
    n, b, mu, std    = h1(tms, bins=10, range=[0,200], stats = True)
    pltl = PlotLabels(x='Time to compute Event Hits (ms)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 3)
    tms = np.array(trt.TimeVoxHits) * 1000
    n, b, mu, std    = h1(tms, bins=10, range=[0,200], stats = True)
    pltl = PlotLabels(x='Time to voxelize hits', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 4)
    tms = np.array(trt.TimeGT) * 1000
    n, b, mu, std    = h1(trt.TimeGT, bins=10, range=[0,200], stats = True)
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
