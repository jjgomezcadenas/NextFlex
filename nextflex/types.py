"""
Defines the types used in NextFlex

"""

import numpy as np
import pandas as pd
import sys, os
from pandas      import DataFrame, Series
from typing      import List, Tuple
from typing      import Union
from typing      import TypeVar
from dataclasses import dataclass

from tics.pd_tics   import get_index_slice_from_multi_index
from tics.util_tics import get_class_name

def repr_base(cls):
    s = f"""<{get_class_name(cls)}>
        Columns = {cls.columns}
        Indexes = {cls.index}
        """
    return s


@dataclass
class McParticles():
    """
    Wrapper data class to give a type to the DataFrame
    representing a collection of particles and events
    with indexes running over particle_id and event_id

    """
    df      : DataFrame

    def __post_init__(self):
        """
        The field columns speciy and thus documents
        the columns expected in the data frame

        """
        self.columns : Tuple[str] = ('particle_name', 'primary', 'mother_id',
               'initial_x', 'initial_y', 'initial_z', 'initial_t',
               'final_x', 'final_y', 'final_z', 'final_t',
               'initial_volume', 'final_volume', 'initial_momentum_x',
               'initial_momentum_y', 'initial_momentum_z', 'final_momentum_x',
               'final_momentum_y', 'final_momentum_z', 'kin_energy', 'length',
               'creator_proc', 'final_proc')

        self.index   : Tuple[str] = ('event_id', 'particle_id')

        assert self.columns == tuple(self.df.columns)
        assert self.index   == tuple(self.df.index.names)


    def event_list(self)->np.array:
        """
        Return an array listing the event ids

        """
        return get_index_slice_from_multi_index(self.df, i = 0)


    def __repr__(self):
        return repr_base(self)


    __str__ = __repr__


@dataclass
class McVertex:
    """
    Wrapper data class to give a type to the DataFrame
    representing the true positions and kinetic energy of the events

    """
    df      : DataFrame
    columns : Tuple[str] = ('x', 'y', 'z', 'kinetic_energy')
    index   : Tuple[str] = ('event_id',)

    def __repr__(self):
        return repr_base(self)

    __str__ = __repr__


@dataclass
class McHits:
    """
    Wrapper data class to give a type to the DataFrame
    representing a collection of monte carlo hits

    """
    df      : DataFrame

    def __post_init__(self):
        """
        The field columns speciy and thus documents
        the columns expected in the data frame

        """
        self.columns : Tuple[str] = ('x', 'y', 'z', 'time', 'energy', 'label')
        self.index   : Tuple[str] = ('event_id', 'particle_id', 'hit_id')

        assert self.columns == tuple(self.df.columns)
        if not self.index == False:
            assert self.index   == tuple(self.df.index.names)


    def event_list(self)->np.array:
        """
        Return an array listing the event ids

        """
        return get_index_slice_from_multi_index(self.df, i = 0)

    def __repr__(self):
        return repr_base(self)

    __str__ = __repr__


@dataclass
class EventHits:
    """
    Wrapper data class to give a type to the DataFrame
    representing a collection of hits. In addition to the
    inner dataframe, the class includes the event_id and two
    parameters:
    - topology  : describes the hits included in this EventHits obect.
        - "all"     : all hits included
        - "primary" : only primary hits included
    - event_type : either "bbonu" (signal) or "1e" (background)

    """
    df         : DataFrame
    event_id   : int
    topology   : str
    event_type : str

    def __post_init__(self):
        """
        The field columns speciy and thus documents the
        columns expected in the data frame

        """
        self.columns : Tuple[str] = ('x', 'y', 'z', 'energy')

        assert self.columns == tuple(self.df.columns)


    def __repr__(self):
        s = f"""<{get_class_name(self)}>
        event number = {self.event_id}
        event type   = {self.event_type}
        topology     = {self.topology}
        Columns = {self.columns}
        """
        return s


    __str__ = __repr__


@dataclass
class EventTrueExtrema:
    """
    Wrapper data class to give a type to the DataFrame
    representing the event true extrema. In addition to the
    inner dataframe, the class includes the event_id and one
    parameters:

    - event_type : either "bbonu" (signal) or "1e" (background)

    """
    df         : DataFrame
    event_id   : int
    event_type : str

    def __post_init__(self):
        """
        The field columns speciy and thus documents the
        columns expected in the data frame

        """
        self.columns : Tuple[str] = ('x', 'y', 'z',
                                     'time', 'energy', 'label')

        assert self.columns == tuple(self.df.columns)


    def __repr__(self):
        s = f"""<{get_class_name(self)}>
        event number = {self.event_id}
        event type   = {self.event_type}
        Columns = {self.columns}
        """
        return s


    __str__ = __repr__


@dataclass
class VoxelHits:
    """
    Wrapper data class to give a type to the DataFrame
    representing a collection of VoxelHits. In addition
    to the inner DataFrame and the event_id
    the class includes three parameters

    - bin_size : describes the size of the voxel
    - baryc    : True if the position of voxel computed from
                 barycenter of the hits, False if computed from
                 the average position of the bins.
    - xyz_bins : defined as:
                 xyz_bins = len(xbins) * len(ybins) * len(zbins) / 1e+6
                 This parameter is proportional to the total number of bins
                 involved in the voxelisation and used to control it.
                 xyz_bins must be at most 1, to avoid memory issues
                 (pd.cut with 10^6 voxels).
    """
    df        : DataFrame
    event_id  : int
    voxel_bin : float
    baryc     : bool
    xyz_bins  : float

    def __post_init__(self):
        """
        The field columns speciy and thus documents the
        columns expected in the data frame

        """
        self.columns : Tuple[str] = ('x', 'y', 'z', 'energy', 'nhits')

        #assert self.columns == tuple(self.df.columns)
        if self.columns != tuple(self.df.columns):
            print(f"error: can't voxelize: df = {self.df}")


    def __repr__(self):
        s = f"""<{get_class_name(self)}>
        event number = {self.event_id}
        voxel bin    = {self.voxel_bin}
        barycenter   = {self.baryc}
        xyz_bins     = {self.xyz_bins}
        Columns      = {self.columns}
        """
        return s

    __str__ = __repr__


@dataclass
class GraphTracks:
    """
    Wrapper data class to give a type to the DataFrame
    representing the reconstructed GTRack obect, plus
    de blobs and true extrema.

    """
    df       : DataFrame

    def __post_init__(self):
        """
        The field columns speciy and thus documents the
        columns expected in the data frame

        """
        self.columns : Tuple[str] = ('event_id', 'track_id',
        'n_voxels', 'trak_length', 'energy',
        'x_e1', 'y_e1', 'z_e1', 'energy_e1', 'nvox_b1', 'energy_b1',
        'x_e2', 'y_e2', 'z_e2', 'energy_e2', 'nvox_b2', 'energy_b2')

        assert self.columns == tuple(self.df.columns)


    def __repr__(self):
        s = f"""<{get_class_name(self)}>
        Columns = {self.columns}
        """
        return s

    __str__ = __repr__
