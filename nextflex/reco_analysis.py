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


def reco_gtrack(ifnames    : List[str],
                topology   : str,
                voxel_bin  : float,
                contiguity : float,
                debug      : bool = False,
                ic         : int = 50)->Tuple[List[GTracks],
                                              TrackRecoStats,
                                              TrackRecoTiming,
                                              TrackRecoEventStats]:
    """
    Driver to reconstruct GraphTracks (or GTracks), including
    statistics and timing book keeping.

    - topology takes values "primary" or "all" and defines whether to consider
    hits from the primary electrons or from all the particles in the event.

    - voxel_bin defines the size of the voxelization cubits (aka voxels)

    - contiguity defines the distance needed for two voxels to be considered
    adyacent

    """

    assert topology == "primary" or topology == "all"

    trs  = TrackRecoStats()
    trt  = TrackRecoTiming()
    tres = TrackRecoEventStats(ifnames, voxel_bin, contiguity)
    GtEvent = []

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

                # Get the Mc Hits in the event. Depending on the topology
                # get the hits corresponding to primary particles or to all
                # particles in the event

                mchits, time = dt_get_event_hits_from_mchits(mcHits,
                                                    event_id     = event_id,
                                                    particle_type = topology)

                trt.TimeEvtHits.append(time)
                trs.NumberMCHits.append(mchits.df.energy.count())
                trs.EnergyMCHits.extend(mchits.df.energy.values)
                trs.TotalEnergyMCHits.append(mchits.df.energy.sum())

                # Voxelize the track in cubits of bin_size
                vox, time = dt_voxelize_hits(mchits,
                                             bin_size = voxel_bin,
                                             baryc    = True)
                vt12, vtinfo = vox
                trt.TimeVoxHits.append(time)
                trt.XyzBins.append(vtinfo.xyz_bins)
                trt.BinSize.append(vtinfo.bin_size)

                vt12df = vt12.df
                voxels = get_voxels_as_list(vt12)

                minimum_d, _ = voxel_distances(voxels)
                trs.MinimumDistVoxels.append(np.max(minimum_d))

                trs.NumberOfVoxels.append(len(voxels))
                trs.VoxelEnergyKeV.append(vt12df.energy.mean()/keV)
                trs.HitsPerVoxel.append(vt12df.nhits.count())

                gtracks, time = dt_make_track_graphs(voxels, contiguity)
                trt.TimeGT.append(time)
                trs.NumberRecTrks.append(len(gtracks))

                if len(gtracks) == 0:
                    print(f" Could not reconstruct any track")

                else:
                    if debug:
                        print(f"number of reco tracks = {len(gtracks)}")

                    GTRKS = [GTrack(gtr, event_id) for gtr in gtracks]

                    if len(gtracks) == 1:
                        tres.e_gt+=1

                GtEvent.append(GTRKS)


    print(f""" Total events analyzed = {tres.e_total},
               Events with a single track = {tres.e_gt}""")

    return GtEvent, trs, tres, trt


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
