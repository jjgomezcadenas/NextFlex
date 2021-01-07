"""
Functions to manipulate MC truth structures

"""

import numpy as np
import pandas as pd
import sys, os

from pandas      import DataFrame, Series
from typing      import List, Tuple
from typing      import Union
from typing      import TypeVar
from dataclasses import dataclass

from invisible_cities.io.mcinfo_io import load_mcparticles_df
from invisible_cities.io.mcinfo_io import load_mchits_df
from tics.system_of_units import *
from tics.pd_tics   import Columns

from tics.pd_tics   import select_on_condition
from tics.pd_tics   import slice_and_select_df
from tics.util_tics import Range

from nextflex.types import McParticles
from nextflex.types import McVertex
from nextflex.types import McHits
from nextflex.types import EventHits



def get_mc_particles(file_name : str)->McParticles:
    """
    Return a McParticles object

    """
    mcParts = load_mcparticles_df(file_name)
    return McParticles(mcParts.sort_index())


def get_mc_primary_particles(mc : McParticles)->McParticles:
    """
    Return primary particles

    """
    return  McParticles(select_on_condition(mc.df, 'primary'))


def get_mc_vertex(mcParts : McParticles)->McVertex:
    """
    Return the true position and the kinetic energy of the events

    """
    # select primary
    mcPrim = select_mc_particles(mcParts, event_slice=slice(None,None),
                                          particle_slice=slice(1,2))

    # Compute the kinetic energies of primary particles for all events
    grouped_multiple = mcPrim.df.groupby(['event_id'])\
                                .agg({'kin_energy': ['sum']})
    grouped_multiple.columns = ['KE']
    KEdf = grouped_multiple.reset_index()
    KE   = 1000. * KEdf.KE.values  # en keV

    #select positions for particle 1 (always primary) and all events
    evt_truePos = slice_and_select_df(mcPrim.df,
                                slices = (slice(None,None), slice(1,1)),
                                columns=['initial_x','initial_y','initial_z'])

    # Remove the 'particle_id' column and rename the fields
    mcVertex = pd.DataFrame(evt_truePos.values,
                            index=evt_truePos.index.droplevel(1),
                            columns = ['x', 'y', 'z'])

    # add the kinetic energy column
    mcVertex['kinetic_energy'] = KE

    # return a wrapped type
    return McVertex(mcVertex)


def select_mc_particles(mc : McParticles,
                        event_slice : slice, particle_slice : slice,
                        columns : Columns = None)->Union[McParticles,
                                                         pd.Series,
                                                         pd.DataFrame]:
    """
    The slice type is of the form slice(start, stop, step).
    Notice that slice(stop) is valid and slice(start, stop) is also valid
    Very importantly, notice that in pandas, the slicing
    follows a different convention than in python
    (which is very convenient), e.g, the slice includes start and stop

    If columns are not selected, the result of the operation is
    a McParticles object, otherwise a
    Series or DataFrame obtained by the column selection.

    """
    if columns == None:
        return McParticles(mc.df.loc[(event_slice, particle_slice), :])
    else:
        return mc.df.loc[(event_slice, particle_slice), columns]


def get_mc_hits(file_name : str)->McHits:
    """
    Return a McHits object

    """
    mc = load_mchits_df(file_name)
    return McHits(mc.sort_index())


def select_mc_hits(mc : McHits,
                        event_slice    : slice,
                        particle_slice : slice,
                        hit_slice      : slice,
                        columns        : Columns = None)->Union[McHits,
                                                                pd.Series,
                                                                pd.DataFrame]:
    """
    The slice type is of the form slice(start, stop, step).
    Notice that slice(stop) is valid and slice(start, stop) is also valid
    Very importantly, notice that in pandas, the slicing
    follows a different convention than in python
    (which is very convenient), e.g, the slice includes start and stop

    If columns are not selected, the result of the operation is
    a McParticles object, otherwise a
    Series or DataFrame obtained by the column selection.

    """
    if columns == None:
        return McHits(mc.df.loc[(event_slice, particle_slice, hit_slice), :])
    else:
        return mc.df.loc[(event_slice, particle_slice, hit_slice), columns]


def total_hit_energy(mc : McHits,
                     event_slice : slice, particle_slice : slice)->pd.DataFrame:
    """
    Selects slices of events and particles and compute the energy of the hits.
    The slice of events allows to define a range of events
    The slice of particles allows the selection of a set of particles.
    For example primary particles with slice(1,2)
    Non primary particles with slice(3, None)
    All particles with slice(None, None)

    """
    mcHitsE = select_mc_hits(mc, event_slice, particle_slice,
                             hit_slice=slice(None, None), columns=['energy'])
    grouped_multiple = mcHitsE.groupby(['event_id']).agg({'energy': ['sum']})
    grouped_multiple.columns = ['total_hit_energy']
    return grouped_multiple.reset_index()


def get_event_hits_from_mchits(mc : McHits,
                               event_id : int,
                               hit_type='all')->EventHits:
    """
    Returns the mchits of and event.
    if hit_type = 'primary' returns hits of primary mc particles only
    else ('all') return hits of all mc particles

    """
    if hit_type == 'primary':
        mchitsL = [select_mc_hits(mc, event_slice=slice(event_id, event_id),
                                  particle_slice =slice(id,id),
                                  hit_slice=slice(None, None)).df\
                                  for id in (1,2)]
        Hits    = pd.concat(mchitsL).reset_index(drop=True)
    else:
        # First get a df with all particles and hits in event_id
        # so that we can extract a list of particles
        mchits = select_mc_hits(mc, event_slice=slice(event_id, event_id),
                                particle_slice =slice(None,None),
                                hit_slice=slice(None, None)).df

        vi = mchits.index.values
        particle_ids = np.unique(list(zip(*vi))[1])
        #print(particle_ids)

        # Now build a list of hits for each particle and then concat
        mchitsL = [select_mc_hits(mc, event_slice=slice(event_id, event_id),
                                  particle_slice =slice(id,id),
                                  hit_slice=slice(None, None)).df\
                                  for id in particle_ids]
        Hits    = pd.concat(mchitsL).reset_index(drop=True)

    eHits = Hits[Hits.label == "ACTIVE"].drop(columns=["time", "label"])

    return EventHits(eHits,event_id)
