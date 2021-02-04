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

from invisible_cities.io.mcinfo_io import load_mcsensor_positions
from invisible_cities.io.mcinfo_io import load_mcsensor_response_df
from invisible_cities.io.mcinfo_io import get_sensor_types
from invisible_cities.io.mcinfo_io import get_sensor_binning
from invisible_cities.io.mcinfo_io import get_event_numbers_in_file

from nextflex.core import NN, NNN
from nextflex.core import Setup, KEY_sensor_fibres, KEY_sensor_pmts
from nextflex.core import get_sensor_response
from nextflex.core import get_sipm_postions


from  tics.stats_tics import bin_data_with_equal_bin_size
from  tics.pd_tics    import get_index_slice_from_multi_index

def event_reco_hits(evt_id         : int,
                    sipm_evt       : pd.DataFrame,
                    sipm_positions : pd.DataFrame,
                    ecut           : float =0):
    """
    Returns a Data Frame of reconstructed SiPM hits

    """
    xHit = []
    yHit = []
    zHit = []
    qHit = []
    sipm = sipm_evt[sipm_evt.charge > ecut]
    sipm_ids = get_index_slice_from_multi_index(sipm, i = 1)

    for sipm_id in sipm_ids:
        xyz = sipm_positions[sipm_positions.sensor_id==sipm_id].iloc[0]
        sipmQZ = sipm.loc[(slice(evt_id, evt_id), slice(sipm_id,sipm_id)), :]
        Q = sipmQZ.charge.values
        Z = sipmQZ.time.values /units.mus
        X = np.ones(len(Q)) * xyz.x
        Y = np.ones(len(Q)) * xyz.y
        xHit.extend(X)
        yHit.extend(Y)
        zHit.extend(Z)
        qHit.extend(Q)

    return pd.DataFrame({'x' : xHit,
                         'y' : yHit,
                         'z' : zHit,
                         'energy'  : qHit})


def tracks_dst(setup      : Setup,
               sipm_ecut  : float = 0,
               record_pmt : bool = False,
               ic         : int   = 100):
    """
    Prepares an analysis dst for tracks. For each event compute:

    1. A list of hits (from SiPMs): x,y,z,Q  in bins of 1 mus.
        sipm_ecut defines the minimum energy of a sipm time bin.
    2. if record_pmt is True compute also the energy of the PMTs.
    3. S2e from MC     --- MC response

    """

    ii = 0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for ifname in setup.ifnames:
            if ii%ic == 0:
                print(f'reading file {ifname}')

            # 1. SiPM response and positions
            sns_response = load_mcsensor_response_df(ifname)
            sns_positions = load_mcsensor_positions(ifname)

            sipm_positions = get_sipm_postions(sns_positions )
            sipm_response = get_sensor_response(sns_response, "SiPM")

            # 2. PMT response
            if record_pmt:
                sipm_response = get_sensor_response(sns_response, "PMT")

            # 3. Events in file
            events = get_index_slice_from_multi_index(sipm_response, i = 0)

            # 4. Loop on event
            Hits =[]
            for event_id in events:
                sipm_evt = sipm_response.loc[(slice(event_id,event_id)), :]

                # 5 loop on sipm_ids:

                hitdf = event_reco_hits(event_id, sipm_evt, sipm_positions,
                                       ecut=sipm_ecut)

                Hits.append(EventHits(hitdf, evemt_id, "all", "data"))

            ii+=1
    return Hits
