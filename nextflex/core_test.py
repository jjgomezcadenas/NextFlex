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
from invisible_cities.io.mcinfo_io import load_mcparticles_df

from nextflex.core import Setup
from nextflex.core import get_evt_true_positions_df


def test_Setup():
    setup = Setup()
    assert setup.name              == 'NEXT-100-PMTs-Masks'
    assert setup.sipmPDE           == 0.4
    assert setup.maskPDE           == 0.4
    assert setup.qTh               == 1.0
    assert setup.maskConfig        == "FLEX100_M6_O6"
    assert setup.mapDIR            == "flexmaps"
    assert setup.fibres            == False
    assert setup.key_sensor_fibres == 100000
    assert setup.key_sensor_pmts   == 100
    assert setup.s1_time           == 1.*units.mus


def test_get_evt_true_positions_df(FDATA):

    test_file  = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Kr83.ACTIVE.1000.next.h5')

    mcParts     = load_mcparticles_df(test_file)
    evt_truePos = get_evt_true_positions_df(mcParts)
    df = evt_truePos.loc[evt_truePos.index[0]]

    print(df)
    assert df.true_x  ==  approx(411.851990)
    assert df.true_y  ==  approx(-83.967979)
    assert df.true_z  ==  approx(981.268127)
