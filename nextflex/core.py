# krypton_dst.py

import os
import sys
import glob
import time
import warnings
import logging

import numpy  as np
import pandas as pd

from pandas import DataFrame, Series
from   dataclasses import dataclass

# Specific IC stuff
import invisible_cities.core.system_of_units  as units
from invisible_cities.core.core_functions     import in_range

from invisible_cities.io.mcinfo_io import load_mcconfiguration
from invisible_cities.io.mcinfo_io import load_mcparticles_df
from invisible_cities.io.mcinfo_io import load_mchits_df
from invisible_cities.io.mcinfo_io import load_mcsensor_positions
from invisible_cities.io.mcinfo_io import load_mcsensor_response_df
from invisible_cities.io.mcinfo_io import get_sensor_types
from invisible_cities.io.mcinfo_io import get_sensor_binning
from invisible_cities.io.mcinfo_io import get_event_numbers_in_file

NN= -999999  # No Number, a trick to aovid nans in data structs


class NNN:
    def __getattr__(self, _):
        return NN


@dataclass
class Setup:
    name              : str   = 'NEXT-100-PMTs-Masks'
    sipmPDE           : float = 0.4
    maskPDE           : float = 0.4
    qTh               : float = 1.0
    maskConfig        : str   = "FLEX100_M6_O6" # means thickness 6 mm hole 6mm
    mapDIR            : str   = "flexmaps" # where to find the SiPM map
    fibres            : bool  = False
    key_sensor_fibres : int   = 100000
    key_sensor_pmts   : int   = 100
    s1_time           : float = 1.*units.mus


def get_evt_true_positions_df(mcParts : DataFrame)->DataFrame:
    """Prepares a DF with the true position of the events using
    the initial vertex of the first particle that is always primary

    """
    evt_truePos = mcParts.loc[pd.IndexSlice[:, 1],
                  ['initial_x', 'initial_y', 'initial_z']]

    # Removing the 'particle_id' column
    evt_truePos = pd.DataFrame(evt_truePos.values,
                               index=evt_truePos.index.droplevel(1),
                               columns = ['true_x', 'true_y', 'true_z'])

    return evt_truePos


def get_evt_true_positions_and_energy(mcParts: DataFrame)->DataFrame:
    """Prepares a DF with the true position of the events using
    the initial vertex of the first particle that is always primary

    """
    grouped_multiple = mcParts.groupby(['event_id']).agg({'kin_energy': ['sum']})
    grouped_multiple.columns = ['KE']
    KEdf = grouped_multiple.reset_index()
    KE   = 1000. * KEdf.KE.values
    evt_truePos = mcParts.loc[pd.IndexSlice[:, 1],
                  ['initial_x', 'initial_y', 'initial_z']]

    # Removing the 'particle_id' column
    evt_truePos = pd.DataFrame(evt_truePos.values,
                               index=evt_truePos.index.droplevel(1),
                               columns = ['true_x', 'true_y', 'true_z'])
    evt_truePos['KE'] = KE

    return evt_truePos
