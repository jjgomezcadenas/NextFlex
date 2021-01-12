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

from  tics.system_of_units import *

from tics.stats_tics    import bin_data_with_equal_bin_size
from tics.pd_tics       import get_index_slice_from_multi_index
from tics.pd_tics       import slice_and_select_df


from nextflex.types import EventHits
from nextflex.types import VoxelHits
from nextflex.types import GraphTracks

from nextflex.core import Setup
from nextflex.reco_functions import Voxel
from nextflex.mctrue_functions import get_mc_hits
from nextflex.mctrue_functions import get_event_hits_from_mchits
from nextflex.reco_functions import voxelize_hits
from nextflex.reco_functions import make_track_graphs
from nextflex.reco_functions import get_voxels_as_list
from nextflex.reco_functions import voxel_distances
from nextflex.reco_functions import GTrack
from nextflex.reco_functions import GTracks
from nextflex.reco_functions import voxels_in_blob
from nextflex.reco_functions import voxel_energy
from nextflex.reco_functions import voxel_nhits
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
    XyzBins     : List[float] = field(default_factory=list)
    BinSize     : List[float] = field(default_factory=list)

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


@dataclass
class TrackRecoAnalysisSetup:
    recoSetup           : Setup
    voxel_bin           : float
    contiguity          : float
    gtracks             : List[GTracks]
    trackRecoStats      : TrackRecoStats
    trackRecoEventStats : TrackRecoEventStats
    trackRecoTiming     : TrackRecoTiming


    def filenames_(self, fileLabel):
        name = f"{fileLabel}_{self.name}"
        return name

    def __post_init__(self):

        self.name =f"{self.recoSetup.name}_" + \
        f"voxel_bin_{self.voxel_bin}_contiguity_{self.contiguity}"
        fileTrackRecoStats       = self.filenames_("TrackRecoStats")
        fileTrackRecoTiming      = self.filenames_("TrackRecoTiming")
        fileTrackRecoEventStats  = self.filenames_("TrackRecoEventStats")
        fileGTracks              = self.filenames_("GTracks")

        self.pathTrackRecoStats       = os.path.join(self.recoSetup.analysis,
                                                     fileTrackRecoStats)
        self.pathTrackRecoTiming      = os.path.join(self.recoSetup.analysis,
                                                     fileTrackRecoTiming)
        self.pathTrackRecoEventStats  = os.path.join(self.recoSetup.analysis,
                                                     fileTrackRecoEventStats)
        self.pathGTracks              = os.path.join(self.recoSetup.analysis,
                                                     fileGTracks)

        self.fileTrackRecoStats       = f"{self.pathTrackRecoStats}.json"
        self.fileTrackRecoTiming      = f"{self.pathTrackRecoTiming}.json"
        self.fileTrackRecoEventStats  = f"{self.pathTrackRecoEventStats}.json"
        self.fileGTracks              = f"{self.pathGTracks}.json"


    def __repr__(self):
        s = f"""
        Analysis Setup      <{self.name}>:
        voxel_bin                    = {self.voxel_bin}
        contiguity                   = {self.contiguity}
        path for TrackRecoStats      = {self.fileTrackRecoStats}
        path for TrackRecoTiming     = {self.fileTrackRecoTiming}
        path for TrackRecoEventStats = {self.fileTrackRecoEventStats}
        path for GTracks             = {self.fileGTracks}

        """
        return s

    def __str__(self):
        return self.__repr__()


def reco_gtrack_from_mc_hits(ifnames    : List[str],
                             voxel_bin  : float,
                             contiguity : float,
                             topology   : str  = "all",
                             event_type : str  = "bb0nu",
                             baryc      : bool = True,
                             debug      : bool = False,
                             ic         : int  = 50)->Tuple[List[GTracks],
                                                           TrackRecoStats,
                                                           TrackRecoTiming,
                                                           TrackRecoEventStats]:
    """
    Driver to reconstruct GraphTracks (or GTracks) from McHits.
    Parameters:
    - ifnames   :  list of input files

    - voxel_bin :  size of the voxelisation

    - contiguity : defines the distance needed for two voxels
                   to be considered adyacent. This quantity is not
                   equal to voxel_bin, since the positions of the voxels
                   is computed from the barycenter (or the average)
                   of the bins they contain, an thus the distance between
                   voxel positions may well be larger than the voxel_bin.
                   The value of contiguity must be chosen to minimise track
                   splits while avoiding absorbing separate tracks into a
                   single reconstructed track. A rule of thumb is to take
                   contiguity as 2 x voxel_bin.
    - topology  :  takes values "primary" or "all" and
                   defines whether to consider
                   hits from the primary electron(s)
                   or from all the particles in the event.

    - event_type : defines whether the event is a bb0nu or a 1e. This is
                   needed to compute the true_extrema of the track.

    - baryc      : Whether to use barycenter of average to determine the
                   voxel position.

    """

    assert topology == "primary" or topology == "all"
    assert event_type == "bb0nu" or event_type == "1e"

    trs  = TrackRecoStats()
    trt  = TrackRecoTiming()
    tres = TrackRecoEventStats(ifnames, voxel_bin, contiguity)
    GtEvent   = []
    tExtrema  = []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for ifname in ifnames:
            tres.f_total+=1

            if tres.f_total % ic == 0:
                print(f'file number = {tres.f_total}, name={ifname}')

            # 1. Get McHits and the event list
            mcHits, time = dt_get_mc_hits(ifname)
            trt.TimeMcHits.append(time)
            events = mcHits.event_list()
            if debug:
                print(f'event list = {events}')

            # 2. Loop over the event list
            for event_id in events:
                tres.e_total+=1
                if debug:
                    print(f'event number = {tres.e_total}')

                # 3. Get the EventHits from the McHits.
                eventHits, time = dt_get_event_hits_from_mchits(mcHits,
                                                                event_id,
                                                                topology,
                                                                event_type)

                # 4. Get the true extrema
                true_extrema = get_true_extrema(eventHits, event_id, event_type)
                tExtrema.append(true_extrema)
                trt.TimeEvtHits.append(time)

                # 5. Voxelize the track in cubits of bin_size
                voxHits, time = dt_voxelize_hits(eventHits, voxel_bin, baryc)
                vt12df = voxHits.df
                trt.TimeVoxHits.append(time)

                # 6. make graph-tracks
                gtracks, time = dt_make_track_graphs(voxHits, contiguity)
                trt.TimeGT.append(time)

                # stats
                trs.NumberMCHits.append(eventHits.df.energy.count())
                trs.EnergyMCHits.append(eventHits.df.energy.mean()/keV)
                trs.TotalEnergyMCHits.append(eventHits.df.energy.sum())
                trt.XyzBins.append(voxHits.xyz_bins)
                trt.BinSize.append(voxHits.bin_size)
                minimum_d, _ = voxel_distances(voxHits)
                trs.MinimumDistVoxels.append(np.max(minimum_d))
                trs.NumberOfVoxels.append(len(vt12df))
                trs.VoxelEnergyKeV.append(vt12df.energy.mean()/keV)
                trs.HitsPerVoxel.append(vt12df.nhits.mean())
                trs.NumberRecTrks.append(len(gtracks))

                if len(gtracks) == 0:
                    print(f" Could not reconstruct any track")

                else:
                    if debug:
                        print(f"number of reco tracks = {len(gtracks)}")

                    GTRKS = [GTrack(gtr, event_id, voxel_bin, contiguity)\ 
                             for gtr in gtracks]

                    if len(gtracks) == 1:
                        tres.e_gt+=1

                GtEvent.append(GTRKS)


    print(f""" Total events analyzed = {tres.e_total},
               Events with a single track = {tres.e_gt}""")

    return GtEvent, tExtrema, trs, tres, trt


def gtrack_df(gtrksEvt : List[List[GTrack]], rb : float)->GraphTracks:
    """
    Output a DataFrame organised by event number and track number
    Compute blob information

    """
    index_tuples = []
    data = []
    for evt_number, gtrks in enumerate(gtrksEvt):
        for trk_number, gt in enumerate(gtrks):
            vb1 = voxels_in_blob(gt, rb, extreme ='e1').df
            vb2 = voxels_in_blob(gt, rb, extreme ='e2').df
            index_tuples.append((evt_number, trk_number))
            data.append({'gtrack_uid': gt.uid,
                         'event_id': gt.event_id,
                         'nvox': len(gt.voxels),
                         'tlength' : gt.length,
                         'x_e1' : gt.extrema['e1'][0],
                         'y_e1' : gt.extrema['e1'][1],
                         'z_e1' : gt.extrema['e1'][2],
                         'energy_e1' : gt.extrema['e1'][3],
                         'nvox_b1': vb1.energy.count(),
                         'energy_b1' : blob_energy(gt, rb, extreme  ='e1'),
                         'x_e2' : gt.extrema['e2'][0],
                         'y_e2' : gt.extrema['e2'][1],
                         'z_e2' : gt.extrema['e2'][2],
                         'energy_e2' : gt.extrema['e2'][3],
                         'nvox_b2': vb2.energy.count(),
                         'energy_b2' : blob_energy(gt, rb, extreme  ='e2'),
                        })
    index = pd.MultiIndex.from_tuples(index_tuples,
                                      names=["evt_number","trk_number"])

    return GraphTracks(pd.DataFrame(data,index))


def event_list_by_multiplicity(gtdf : GraphTracks)->Tuple[List[int], List[int]]:
    """
    Return two event lists:
    1. Events with single tracks (EST) list
    2. Events with multiple tracks (EMT) list

    """
    df = gtdf.df
    event_list = get_index_slice_from_multi_index(df,0)
    EST = []
    EMT = []

    for event_number in event_list:
        # slice event_number
        evt = slice_and_select_df(df,
                                  slices = (slice(event_number,event_number),
                                            slice(None,None)),
                                  columns = ['nvox'])
        if len(evt) == 1:
            EST.append(event_number)
        else:
            EMT.append(event_number)
    return EST, EMT


def select_tracks_by_multiplicity(gtrks   : GraphTracks,
                                  trkList : List[int])->pd.DataFrame:
    gtdf = gtrks.df
    cg = [x for x in gtdf.columns.values]
    gtevt = [slice_and_select_df(gtdf,
                             slices = (slice(event_id,event_id),
                                       slice(None,None)),
                             columns = cg) for event_id in trkList]

    return pd.concat(gtevt)


def select_gtrack_topology(gtrks : GraphTracks,
                           topology : str ="single")->pd.DataFrame:
    """
    Take a GraphTracks object and return a DataFrame
    with selected objects by topology:
        single -- single tracks
        multi  -- multiple tracks

    """
    st, mt = event_list_by_multiplicity(gtrks)

    if topology == "single":
        #gt   = select_tracks_by_multiplicity(gtrks, st)
        #gt1t = pd.DataFrame(gt.values,
        #                     index=gt.index.droplevel(1),
        #                     columns = gt.columns)
        # return gt1t
        return select_tracks_by_multiplicity(gtrks, st)
    else:
        return select_tracks_by_multiplicity(gtrks, mt)


def distance_between_extrema(df : DataFrame)->DataFrame:
    """
    Compute the distance between the extremes of tracks

    """
    def compute_distances(df, event_list, extreme, DE):
        if extreme == "e1":
            columns = ["nvox", "x_e1", "y_e1", "z_e1"]
            c1      = ["x_e1", "y_e1", "z_e1"]
        else:
            columns = ["nvox", "x_e2", "y_e2", "z_e2"]
            c1      = ["x_e2", "y_e2", "z_e2"]

        for event_number in event_list:
            # slice event_number
            evt = slice_and_select_df(df,
                                      slices = (slice(event_number,event_number),
                                                slice(None,None)),
                                      columns = columns)
            # drop event_number index
            evt = pd.DataFrame(evt.values,
                               index=evt.index.droplevel(0),
                               columns = columns)

            # replace the actuval values of x,y,z by the differences
            # (xi-x0) (i=0,1...) same or y and z
            for c in c1:
                evt[c] =  evt.loc[:, c] - evt.at[0, c]

            # compute the euclidean distance (of the differences):
            # first row is zeros
            de = evt[c1].apply(lambda x : np.linalg.norm(x) , axis = 1 )
            DE.extend(de.values[1:])

    DE1 = []
    DE2 = []
    event_list = get_index_slice_from_multi_index(df,0)
    compute_distances(df, event_list, 'e1', DE1)
    compute_distances(df, event_list, 'e2', DE2)
    data ={'distances_e1' : DE1, 'distances_e2' : DE2}
    return pd.DataFrame(data)


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


def gtrack_summary(gts : GtrkStats, trk_number : int):
    print(f"""
    GTrack number = {trk_number}
    NumberOfVoxels    = {gts.NumberOfVoxels[trk_number]}
    TrackLength       = {gts.TrackLength[trk_number]}
    NumberVoxelsBlob1 = {gts.NumberVoxelsBlob1[trk_number]}
    NumberVoxelsBlob2 = {gts.NumberVoxelsBlob2[trk_number]}
    EnergyBlob1       = {gts.EnergyBlob1[trk_number]}
    EnergyBlob2       = {gts.EnergyBlob2[trk_number]}
    """)
