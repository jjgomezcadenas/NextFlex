"""
Tests for core functions
"""
import os
import pytest
import numpy  as np
import tables as tb
from pytest           import approx

import numpy          as np
import pandas as pd

import invisible_cities.core.system_of_units  as units
from invisible_cities.core.core_functions     import in_range

from invisible_cities.io.mcinfo_io import load_mcparticles_df

from nextflex.core import Setup
from nextflex.core import get_evt_true_positions_df
from nextflex.core import get_evt_true_positions_and_energy
from nextflex.core import get_sensor_response
from nextflex.core import sensor_response_ti
from nextflex.core import event_sensor_response_ti
from nextflex.core import sensor_number_response_ti
from nextflex.core import mcparts_and_sensors_response
#from nextflex.krypton_dst import prepare_tmpdir
#from nextflex.krypton_dst import collect_h5files
from nextflex.krypton_dst import get_file_name


def test_get_file_name(FDATA):
    setup   = Setup(FDATA)
    ifname  = 'FLEX100_M6_O6.Kr83.ACTIVE.1000.next.h5'
    csvname = 'FLEX100_M6_O6.Kr83.ACTIVE.1000.next.csv'
    path    = os.path.join(FDATA,"testData")
    file    = get_file_name(ifname, setup)
    assert file == f"{FDATA}/{setup.tpConfig}/{setup.name}/{csvname}"
