# krypton_dst.py

import os
import sys
import glob
import time
import warnings
import logging

import numpy  as np
import pandas as pd
from   dataclasses import dataclass

from pandas import DataFrame, Series
from typing import List, Tuple
from typing import Union

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

from nextflex.core import NN, NNN
from nextflex.core import Setup
from nextflex.core import get_evt_true_positions_df
from nextflex.core import get_evt_true_positions_and_energy
from nextflex.core import get_sensor_response
from nextflex.core import sensor_response_ti
from nextflex.core import mcparts_and_sensors_response
from nextflex.core import get_s1
from nextflex.core import get_s2
from nextflex.core import get_qtot
from nextflex.core import get_position
from nextflex.core import diff_pos


def get_krdst(dsts     : List[DataFrame],
             sipm_map : DataFrame,
             setup    : Setup):

    mcParts, energy_sensors_response, sipm_response = dsts
    sipmdf       = sensor_response_ti(sipm_response)

    krdf         = get_evt_true_positions_and_energy(mcParts)
    krdf['S1e']  = get_s1(energy_sensors_response)
    krdf['S2e']  = get_s2(energy_sensors_response)
    krdf['Qtot'] = get_qtot(sipmdf, setup)

    pq   = get_position(krdf.index, sipmdf, sipm_map, setup)
    dxdf = diff_pos(krdf, pq)

    krdst = pd.concat([krdf, pq, dxdf], axis=1)

    return krdst


def kr_dst(ifnames           : List[str],
           sipm_map          : DataFrame,
           setup             : Setup,
           ic                : int   = 100):
    """
    Prepares an analysis dst for Krypton, including:

    1. True positions from the MC
    2. Computed positions from Barycenter (around SiPM with max charge) --
       after integrating all time bins
    3. S1e from fibers/PMT --- MC response
    4. S2e from MC     --- MC response
    5. charge in the SiPMs (max, left, right, up, down)

    """

    def get_file_name(ifname):
        lname = ifname.split('.')
        t1 = ".".join(lname[1:-1])
        t = "".join([lname[0],t1])
        f =f"{t}.csv"
        return f


    GF =[]
    BF =[]
    ii = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for ifname in ifnames:
            if ii%ic == 0:
                print(f'reading file {ifname}')
            dsts = mcparts_and_sensors_response(ifname, setup)
            if dsts == False:
                BF.append(ifname)
                continue
            else:
                GF.append(ifname)

                krdst = get_krdst(dsts, sipm_map, setup)
                file  = get_file_name(ifname)
                if ii%ic == 0:
                    print(f'saving file {file}, with {len(krdst.index)} events')
                krdst.to_csv(file)
            ii+=1
    return GF, BF


def kr_join_dst(ifnames, verbose=False, ic=100):
    """
    Joins the csv dst files
    """

    krdst           = pd.DataFrame()
    BF =[]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for ifname in ifnames:
            if verbose:
                print(f'reading file {ifname}')
            try:
                kdst = pd.read_csv(ifname)
            except:
                print(f'Failed reading csv dst file ={ifname}')
                BF.append(ifname)
                continue

            if verbose:
                print(kdst)
            krdst = krdst.append(kdst)

    return krdst, BF
