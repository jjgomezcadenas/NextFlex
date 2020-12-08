import os
import pytest
import numpy  as np
import pandas as pd
from   dataclasses import dataclass

from pandas      import DataFrame
from invisible_cities.io.mcinfo_io import load_mcparticles_df
from invisible_cities.io.mcinfo_io import load_mcsensor_response_df

@dataclass
class TestSetup:
    mcParts           : DataFrame
    sns_response_pmt  : DataFrame
    sns_response_fib  : DataFrame
    sipm_map          : DataFrame


@pytest.fixture(scope = 'session')
def FDIR():
    return os.environ['NEXTFLEX']


@pytest.fixture(scope = 'session')
def FDATA():
    return os.environ['FLEXDATA']


@pytest.fixture(scope='session')
def mc_sns_sipm_map(FDATA):
    testFile      = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Kr83.ACTIVE.1000.next.h5')
    testFile2 = os.path.join(FDATA,"testData",
                                'NEXT_FLEX_Fibres_M6_06.Kr83.ACTIVE.1.next.h5')
    mapFile       = os.path.join(FDATA,"testData", 'sipm_map.csv')
    mcParts       = load_mcparticles_df(testFile)
    sipm_map      = pd.read_csv(mapFile)
    sns_response_pmt  = load_mcsensor_response_df(testFile)
    sns_response_fib  = load_mcsensor_response_df(testFile2)
    return TestSetup(mcParts, sns_response_pmt, sns_response_fib, sipm_map)
