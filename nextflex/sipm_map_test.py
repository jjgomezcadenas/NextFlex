# krypton_dst.py

import os
import sys
import glob
import time
import warnings

from pandas import DataFrame

import numpy  as np
import pandas as pd


import invisible_cities.core.system_of_units  as units
from invisible_cities.core.core_functions     import in_range

from nextflex.sipm_map import sipm_pos
from nextflex.core import find_pitch
from nextflex.core import NN


def test_sipm_pos(FDATA, mc_sns_sipm_map):

    testFile      = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Kr83.ACTIVE.1000.next.h5')

    XL = mc_sns_sipm_map.sipm_map.xl.values
    XR = mc_sns_sipm_map.sipm_map.xr.values
    YU = mc_sns_sipm_map.sipm_map.yu.values
    YD = mc_sns_sipm_map.sipm_map.yd.values

    pitch = find_pitch(testFile)

    P = np.array(len(XL) *[2 * pitch])
    DX = [xr - xl if xl != NN and xr != NN else 2 * pitch for xl,xr in zip(XL, XR) ]
    DY = [yu - yd if yu != NN and yd != NN else 2 * pitch for yu,yd in zip(YU, YD) ]
    assert np.allclose(DX, P, rtol=1e-03, atol=1e-03)
    assert np.allclose(DY, P, rtol=1e-03, atol=1e-03)


def test_sipm_indx(mc_sns_sipm_map):
    """
    Checks that the indexes match along one direction
    (X --- indexes always jump in units one one)

    """
    XL = mc_sns_sipm_map.sipm_map.xl.values
    XR = mc_sns_sipm_map.sipm_map.xr.values
    YU = mc_sns_sipm_map.sipm_map.yu.values
    YD = mc_sns_sipm_map.sipm_map.yd.values

    DX = [xr - xl if xl != NN and xr != NN else 2 for xl,xr in zip(XL, XR) ]
    t1 = np.allclose(DX, 2, rtol=1e-03, atol=1e-03)
    return t1
