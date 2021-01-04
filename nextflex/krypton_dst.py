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

    # integrate time for the SiPMs.
    sipmdf       = sensor_response_ti(sipm_response)

    krdf         = get_evt_true_positions_and_energy(mcParts)
    krdf['S1e']  = get_s1(energy_sensors_response)
    krdf['S2e']  = get_s2(energy_sensors_response)
    # krdf['Qtot'] = get_qtot(sipmdf, setup)

    pq   = get_position(krdf.index, sipmdf, sipm_map, setup)
    dxdf = diff_pos(krdf, pq)

    krdst = pd.concat([krdf, pq, dxdf], axis=1)

    return krdst


def kr_dst(sipm_map          : DataFrame,
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

    GF =[]
    BF =[]
    ii = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for ifname in setup.ifnames:
            if ii%ic == 0:
                print(f'reading file {ifname}')
            dsts = mcparts_and_sensors_response(ifname, setup)
            if dsts == False:
                BF.append(ifname)
                continue
            else:
                GF.append(ifname)

                krdst = get_krdst(dsts, sipm_map, setup)
                file  = get_file_name(ifname, setup)
                if ii%ic == 0:
                    print(f'saving file {file}, with {len(krdst.index)} events')
                krdst.to_csv(f"{file}")
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


def get_file_name(ifname, setup):
    """Get the csv file names from h5 file names"""
    xname  = ifname.split("/")
    h5 = xname[-1]
    h5name = h5.split(".")
    fs = ".".join(h5name[0:-1])
    fcsv =f"{fs}.csv"
    #tmpdir = f"{FDATA}/{setup.tpConfig}/{setup.name}"
    file = f"{setup.tmpdir}/{fcsv}"

    return file


def prepare_tmpdir(setup):
    """Prepare a temporary directory to store cvs files """
    #tmpdir = f"{setup.iPATH}/{setup.name}"
    if not os.path.exists(setup.tmpdir):
        print(f"creating dir {setup.tmpdir}")
        os.makedirs(setup.tmpdir)
    else:
        print(f"cleaning up .csv files from  dir {tmpdir}")
        try:
            os.system(f'rm -r {setup.tmpdir}/*.csv')
        except:
            print("No csv files found in directory")


# def collect_h5files(FDATA, setup):
#     """Collects h5 files to run kr_dst over them"""
#
#     ddir = f"{FDATA}/{setup.tpConfig}"
#     ifnames = glob.glob(f"{ddir}/*.h5")
#     return ifnames

# def collect_csvfiles(setup):
#     """Collects csv files to run kr_dst over them"""
#
#     ddir = f"{FDATA}/{setup.tpConfig}/{setup.name}"
#     ifnames = glob.glob(f"{ddir}/*.csv")
#     return ifnames

# # collect csv files
# ifnames2 = glob.glob(f"{FDATA}/{setup.tpConfig}/*.csv")
