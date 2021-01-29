import numpy as np
import pandas as pd
import networkx as nx
import json
import os
import uuid
from dataclasses import dataclass
from dataclasses import field
from dataclasses import asdict

from itertools   import combinations
from  tics.system_of_units import *
from  tics.stats_tics import bin_data_with_equal_bin_size

from pandas      import DataFrame
from typing      import List, Tuple, Dict
from typing      import TypeVar

from nextflex.types import EventHits
from nextflex.types import VoxelHits

Voxel = TypeVar('Voxel', Tuple[float], np.array)  # define voxel as type
#
# idx = pd.IndexSlice

@dataclass
class GTrack:
    """
    A graph-track
    gt       : a nx Graph representing the track
    event_id : event number
    uid      : A unique identifier of the track

    """
    gt         : nx.Graph
    event_id   : int
    voxel_bin  : float
    contiguity : float


    def __post_init__(self):
        #self.uid         = uuid.uuid1()
        self.voxels     = self.gt.nodes()
        self.voxels_df  = pd.DataFrame(self.voxels,
                                       columns=['x','y','z','energy', 'nhits'])
        self.extrema    = {}
        self.distances  = shortest_paths(self.gt)
        #extrema 1 and 2
        e1, e2, self.length = find_extrema_and_length_from_dict(self.distances)
        self.extrema['e1'] = e1
        self.extrema['e2'] = e2
        p1 = pd.DataFrame(self.extrema).T
        p1.columns=['x','y','z','e', 'nhits']
        self.extrema_df = p1

    def __repr__(self):
        s = f"""
        <GTrack>:
        event_id           = {self.event_id}
        voxel bin          = {self.voxel_bin}
        contiguity         = {self.contiguity}
        number of voxels   = {len(self.voxels)}
        extrema voxels: e1 = {self.extrema['e1']},
                        e2 = {self.extrema['e2']}
        track length:      = {self.length} mm
        """
        return s

    __str__ = __repr__


@dataclass
class GTracks:
    """
    A container of GTrack
    """
    gtracks         : List[GTrack] = field(default_factory=list)


def voxelize_hits(hits     : EventHits,
                  bin_size : int,
                  baryc    : bool = True)->VoxelHits:
    """
    Takes a EventHits objects wit fields (x,y,z,energy)
    voxelize the data in cubic voxels of size bin_size and return a
    VoxelHits object.

    """

    def voxelize_hits_bc(df : pd.DataFrame)->pd.Series:
        """
        Computes the barycenters in x,y,z

        """
        def barycenter(df, var, etot):
            return np.sum([np.dot(a,b)\
                           for a, b  in zip(df[var] , df.energy)]) / etot
        d = {}
        etot   = df['energy'].sum()
        d['x'] = barycenter(df, 'x', etot)
        d['y'] = barycenter(df, 'y', etot)
        d['z'] = barycenter(df, 'z', etot)
        d['energy'] = etot
        d['nhits'] = df['energy'].count()
        return pd.Series(d)

    def voxelize_hits_mean(df : pd.DataFrame)->pd.Series:
        """
        Compute the averages in x, y, z

        """
        d = {}
        d['x'] = df['x'].mean()
        d['y'] = df['y'].mean()
        d['z'] = df['z'].mean()
        d['energy'] = df['energy'].sum()
        d['nhits'] = df['energy'].count()
        return pd.Series(d)

    def check_xyz_bins(xbins, ybins, zbins, bin_size):
        """
        Get sure that xyz_bins <= 1e+6 otherwise
        rebin voxels

        """
        xyz_bins = len(xbins) * len(ybins) * len(zbins) / 1e+6
        if xyz_bins > 1:
            print(f'xyz_bins = {xyz_bins} is too large, try larger bin_size')
            while xyz_bins > 1:
                bin_size+=1
                xbins, ybins, zbins = bin_data_with_equal_bin_size([df.x,
                                                                    df.y,
                                                                    df.z],
                                                                    bin_size)
                xyz_bins = len(xbins) * len(ybins) * len(zbins) / 1e+6
            print(f' new bin_size = {bin_size} xyz_bins = {xyz_bins} ')
        return xyz_bins, bin_size, xbins, ybins, zbins

    df = hits.df.copy()
    xbins, ybins, zbins = bin_data_with_equal_bin_size([df.x, df.y, df.z],
                                                        bin_size)

    xyz_bins, bin_size, xbins, ybins, zbins,  = check_xyz_bins(xbins,
                                                               ybins,
                                                               zbins,
                                                               bin_size)

    df['x_bins'] = pd.cut(df['x'],bins=xbins, labels=range(len(xbins)-1))
    df['y_bins'] = pd.cut(df['y'],bins=ybins, labels=range(len(ybins)-1))
    df['z_bins'] = pd.cut(df['z'],bins=zbins, labels=range(len(zbins)-1))

    if baryc:
        vhits = df.groupby(['x_bins','y_bins','z_bins'])\
                                             .apply(voxelize_hits_bc)\
                                             .dropna().reset_index(drop=True)
    else:
        vhits = df.groupby(['x_bins','y_bins','z_bins'])\
                                             .apply(voxelize_hits_mean)\
                                             .dropna().reset_index(drop=True)

    return VoxelHits(vhits, hits.event_id, bin_size, baryc, xyz_bins)


def get_voxels_as_list(voxelHits : VoxelHits)->List[Voxel]:
    """
    Return the voxels as a list of tuples with the coordinates
    (this is needed for networkx)

    """
    voxeldf = voxelHits.df
    return [(v[0], v[1], v[2], v[3], v[4]) for v in voxeldf.values]


def voxel_position(v : Voxel)->np.array:
    """
    Return the position of a voxel as a numpy array

    """
    return np.array([v[0], v[1], v[2]])


def voxel_energy(v : Voxel)->float:
    """
    Return the energy of a voxel

    """
    return v[3]


def voxel_nhits(v : Voxel)->float:
    """
    Return the number of hits of a voxel

    """
    return v[4]


def distance_between_two_voxels(va : Voxel, vb : Voxel)->float:
    """
    Return the distance between two voxels

    """
    return np.linalg.norm(voxel_position(vb) - voxel_position(va))


def voxel_distances(voxelHits : VoxelHits)->Tuple[np.array]:
    """
    Return a numpy array with the distance (inclusive) between any pair
    of voxels, and another array with the minimum distance between a
    voxel and all the others.

    """
    voxels = get_voxels_as_list(voxelHits)
    DSTM = []
    DSTI = []
    for va in voxels:
        DST = []
        for vb in voxels:
            if not np.allclose(va,vb):
                DST.append(distance_between_two_voxels(va,vb))
                DSTI.append(distance_between_two_voxels(va,vb))
        if len(DST) > 0:
            DSTM.append(np.min(DST))
    return np.array(DSTM), np.array(DSTI)


def voxel_distance_pairs(voxels : List[Voxel])->Tuple[np.array]:
    """
    Return a numpy array with the distance (inclusive) between any pair
    of voxels, use a list conprehension.

    """
    return np.array([distance_between_two_voxels(va, vb)\
                     for va, vb in combinations(voxels, 2)])


def make_track_graphs(voxelHits  : VoxelHits,
                      contiguity : float)->List[nx.Graph]:
    """
    Make "graph-tracks" (gtracks) using networkx:

    1. Define a graph such that each voxel is a node and there is a link
    (or edge) between each pair of nodes which are at a distance smaller
    than defined by contiguity.

    2. Return a list of graphs made with connected components. Each connected
    component graph is made of a set of connected nodes (eg. nodes which are
    at a distance smaller than contiguity)

    """
    def connected_component_subgraphs(G):
        return (G.subgraph(c).copy() for c in nx.connected_components(G))


    voxels = get_voxels_as_list(voxelHits)
    voxel_graph = nx.Graph()
    voxel_graph.add_nodes_from(voxels)
    for va, vb in combinations(voxels, 2):
        d = distance_between_two_voxels(va, vb)
        if d < contiguity:
            voxel_graph.add_edge(va, vb, distance = d)
    return list(connected_component_subgraphs(voxel_graph))


def gtrack_voxels(gtrack : nx.Graph)->pd.DataFrame:
    """
    Return a DataFrame of voxels from a gtrack

    """
    return pd.DataFrame(gtrack.nodes(), columns=['x','y','z','energy', 'nhits'])


def find_extrema_and_length_from_dict(distance : Dict[Voxel,
                        Dict[Voxel, float]]) -> Tuple[Voxel, Voxel, float]:
    """
    Find the extrema and the length of a track,
    given its dictionary of distances.

    """
    if not distance:
        raise NoVoxels
    if len(distance) == 1:
        only_voxel = next(iter(distance))
        return (only_voxel, only_voxel, 0.)
    first, last, max_distance = None, None, 0
    for (voxel1, dist_from_voxel_1_to), (voxel2, _) in \
                                        combinations(distance.items(), 2):
        d = dist_from_voxel_1_to[voxel2]
        if d > max_distance:
            first, last, max_distance = voxel1, voxel2, d
    return first, last, max_distance


def shortest_paths(track_graph : nx.Graph) -> Dict[Voxel, Dict[Voxel, float]]:
    """
    Compute shortest path lengths between all nodes in a weighted graph.

    """
    return dict(nx.all_pairs_dijkstra_path_length(track_graph,
                                                  weight='distance'))


def find_extrema_and_length(track: nx.Graph) -> Tuple[Voxel, Voxel, float]:
    """
    Find the pair of voxels separated by the greatest geometric
    distance along the track.

    """
    distances = shortest_paths(track)
    e_a, e_b, tl = find_extrema_and_length_from_dict(distances)
    return e_a, e_b, tl


def voxels_in_blob(gt : GTrack, rb : float, extreme : str ='e1')->pd.DataFrame:
    """
    Return the a VoxelHits object with the voxels contained in a blob
    of radius rb around the extreme.

    """
    distances_from_extreme = gt.distances[gt.extrema[extreme]]
    vb = [voxel for voxel, distance in distances_from_extreme.items()\
          if distance <= rb]

    return pd.DataFrame(vb,
                     columns =['x', 'y', 'z', 'energy', 'nhits'])


def blob_energy(gt : GTrack, rb : float, extreme : str ='e1', unit=keV)->float:
    """
    Return the total energy contained in a blob of radius rb around
    the extreme.

    """
    voxels = voxels_in_blob(gt, rb, extreme)
    return voxels.energy.sum() / unit
