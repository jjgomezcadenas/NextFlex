import numpy as np
import pandas as pd
import networkx as nx
import os

import warnings
import functools
import time

from dataclasses import dataclass
from dataclasses import field
from dataclasses import asdict

from pandas      import DataFrame
from typing      import List, Tuple, Dict
from typing      import TypeVar

import matplotlib.pyplot as plt

from  tics.system_of_units import *
from  tics.stats_tics import bin_data_with_equal_bin_size
from tics.histograms    import h1
from tics.histograms    import PlotLabels
from tics.histograms    import plot_histo

from nextflex.types import EventHits
from nextflex.types import VoxelHits
from nextflex.reco_functions import Voxel
from nextflex.mctrue_functions import get_mc_hits
from nextflex.mctrue_functions import get_event_hits_from_mchits
from nextflex.reco_functions import voxelize_hits
from nextflex.reco_functions import make_track_graphs
from nextflex.reco_functions import get_voxels_as_list
from nextflex.reco_functions import voxel_distances
from nextflex.reco_functions import GTrack
from nextflex.reco_functions import voxels_in_blob
from nextflex.reco_functions import blob_energy
import json


def decorator_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        #print(f"Starting {func.__name__!r}")
        start_time = time.perf_counter()    # 1
        value      = func(*args, **kwargs)
        end_time   = time.perf_counter()      # 2
        run_time   = end_time - start_time    # 3
        #print(f"Executed {func.__name__!r} in {run_time:.4f} secs")
        return value, run_time

    return wrapper


dt_get_mc_hits = decorator_timer(get_mc_hits)
dt_get_event_hits_from_mchits = decorator_timer(get_event_hits_from_mchits)
dt_voxelize_hits = decorator_timer(voxelize_hits)
dt_make_track_graphs = decorator_timer(make_track_graphs)


@dataclass
class TrackRecoStats:
    NumberMCHits         : List[float] = field(default_factory=list)
    EnergyMCHits         : List[float] = field(default_factory=list)
    TotalEnergyMCHits    : List[float] = field(default_factory=list)
    NumberOfVoxels       : List[float] = field(default_factory=list)
    MinimumDistVoxels    : List[float] = field(default_factory=list)
    VoxelEnergyKeV       : List[float] = field(default_factory=list)
    HitsPerVoxel         : List[float] = field(default_factory=list)
    NumberRecTrks        : List[float] = field(default_factory=list)


@dataclass
class TrackRecoTiming:
    TimeMcHits  : List[float] = field(default_factory=list)
    TimeEvtHits : List[float] = field(default_factory=list)
    TimeVoxHits : List[float] = field(default_factory=list)
    TimeGT      : List[float] = field(default_factory=list)


@dataclass
class TrackRecoEventStats:
    ifnames    : List[str]
    voxel_bin  : int
    contiguity : float
    f_total    : int = 0
    e_total    : int = 0
    e_gt       : int = 0

    def __repr__(self):
        s = f"""
        <Track Reconstruction Statistics>
        first file analyzed        = {self.ifnames[0]}
        number of files analyzed   = {self.f_total}
        size of voxel bins         = {self.voxel_bin}
        contiguity parameter       = {self.contiguity}
        Number of events analyized = {self.e_total}
        Number of events 1 GT      = {self.e_gt}
        """
        return s

    __str__ = __repr__


@dataclass
class GtrkStats:
    NumberOfVoxels    : List[float] = field(default_factory=list)
    TrackLength       : List[float] = field(default_factory=list)
    NumberVoxelsBlob1 : List[float] = field(default_factory=list)
    NumberVoxelsBlob2 : List[float] = field(default_factory=list)
    EnergyBlob1       : List[float] = field(default_factory=list)
    EnergyBlob2       : List[float] = field(default_factory=list)


def save_to_JSON(acls, path, numpy_convert=True):
    """
    Converts reco analysis data classes to JSON and writes it to path
    numpy_convert is True when the data needs to be converted
    from type numpy (this happens for lists extracted from a DF)

    """

    dcls = asdict(acls)
    if numpy_convert:
        facls = {k:[float(x) for x in v] for (k,v) in dcls.items()}
    else:
        facls = dcls

    # then write
    with open(path, 'w') as fp:
        json.dump(facls, fp)


def load_from_JSON(path):
    """
    Reads a JSON object as a dict

    """
    with open(path) as json_file:
        jdict = json.load(json_file)
    return jdict


def reco_gtrack(ifnames, voxel_bin = 1, contiguity=2, debug=False, ic = 50):
    """
    Driver to reconstruct GraphTracks (or GTracks), including
    statistics and timing book keeping.

    """

    trs  = TrackRecoStats()
    trt  = TrackRecoTiming()
    tres = TrackRecoEventStats(ifnames, voxel_bin, contiguity)

    GTRKS = []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for ifname in ifnames:
            tres.f_total+=1

            if tres.f_total % ic == 0:
                print(f'file number = {tres.f_total}, name={ifname}')
            mcHits, time = dt_get_mc_hits(ifname)
            trt.TimeMcHits.append(time)

            events = mcHits.event_list()
            if debug:
                print(f'event list = {events}')

            for event_id in events:
                tres.e_total+=1
                if debug:
                    print(f'event number = {tres.e_total}')

                mchits, time = dt_get_event_hits_from_mchits(mcHits,
                                                             event_id=event_id,
                                                             hit_type='all')
                trt.TimeEvtHits.append(time)
                trs.NumberMCHits.append(mchits.df.energy.count())
                trs.EnergyMCHits.extend(mchits.df.energy.values)
                trs.TotalEnergyMCHits.append(mchits.df.energy.sum())

                vt12, time = dt_voxelize_hits(mchits, bin_size = voxel_bin,
                                              baryc = True)
                trt.TimeVoxHits.append(time)

                vt12df = vt12.df
                voxels = get_voxels_as_list(vt12)
                minimum_d, _ = voxel_distances(voxels)
                trs.MinimumDistVoxels.extend(minimum_d)

                trs.NumberOfVoxels.append(len(voxels))
                trs.VoxelEnergyKeV.extend(vt12df.energy/keV)
                trs.HitsPerVoxel.extend(vt12df.nhits)

                gtracks, time = dt_make_track_graphs(voxels, contiguity)
                trt.TimeGT.append(time)
                trs.NumberRecTrks.append(len(gtracks))

                if len(gtracks) == 0:
                    print(f" Could not reconstruct any track")
                    continue

                elif len(gtracks) >1:
                    if debug:
                        print(f"number of reco tracks = {len(gtracks)}")
                    continue

                else:
                    if debug:
                        print(f" One track!!!")
                    gtrack = gtracks[0]
                    GTRKS.append(GTrack(gtrack, event_id))
                    tres.e_gt+=1

    print(f""" Total events analyzed = {tres.e_total},
               Events with a single track = {tres.e_gt}""")

    return GTRKS, trs, tres, trt


def reco_gtrack_blobs(gtrks : List[nx.Graph], rb = 10):
    gs = GtrkStats()
    for gtrk in gtrks:

        gs.NumberOfVoxels.append(len(gtrk.voxels))
        gs.TrackLength.append(gtrk.length)
        vb1 = voxels_in_blob(gtrk, rb, extreme ='e1').df
        #print(vb1)
        gs.NumberVoxelsBlob1.append(vb1.energy.count())
        vb2 = voxels_in_blob(gtrk, rb, extreme ='e2').df
        gs.NumberVoxelsBlob2.append(vb2.energy.count())
        gs.EnergyBlob1.append(blob_energy(gtrk, rb, extreme  ='e1'))
        gs.EnergyBlob2.append(blob_energy(gtrk, rb, extreme  ='e2'))
    return gs


def reco_gtrack_stats_histos(trs : TrackRecoStats, figsize=(14,10)):
    """
    Plots the following histograms

    1. NumberMCHits
    2. EnergyMCHits
    3. NumberOfVoxels
    4. VoxelEnergyKeV
    5. HitsPerVoxel
    6. MinimumDistVoxels
    7. NumberRecTrks

    """
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(4, 2, 1)
    n, b, mu, std    = h1(trs.NumberMCHits, bins=10, range=[0,1500], stats = True)
    pltl = PlotLabels(x='Number of MC Hits', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 2)
    energyKeV = np.array(trs.EnergyMCHits) / keV
    n, b, mu, std    = h1(energyKeV, bins=10, range=[0,25], stats = True)
    pltl = PlotLabels(x='Energy Hits (keV)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 3)
    energyKeV = np.array(trs.TotalEnergyMCHits) / keV
    n, b, mu, std    = h1(energyKeV, bins=10, range=[0,2500], stats = True)
    pltl = PlotLabels(x='Total energy (keV)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)


    ax      = fig.add_subplot(4, 2, 4)
    n, b, mu, std    = h1(trs.NumberOfVoxels, bins=10, range=[0,50], stats = True)
    pltl = PlotLabels(x='number of voxels', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 5)
    n, b, mu, std    = h1(trs.VoxelEnergyKeV, bins=10, range=[0,1000], stats = True)
    pltl = PlotLabels(x='Voxel energy (keV)', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 6)
    n, b, mu, std    = h1(trs.HitsPerVoxel, bins=10, range=[0,300], stats = True)
    pltl = PlotLabels(x='Hits per voxel', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 7)
    n, b, mu, std    = h1(trs.MinimumDistVoxels, bins=10, range=[0,20], stats = True)
    pltl = PlotLabels(x='minimum distance between voxels', y='events', title=None)
    plot_histo(pltl, ax, legend=True)

    ax      = fig.add_subplot(4, 2, 8)
    n, b, mu, std    = h1(trs.NumberRecTrks, bins=10, range=[0,10], stats = True)
    pltl = PlotLabels(x='Number of reconstructed tracks', y='events', title=None)
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
