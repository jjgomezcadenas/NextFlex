import os
import pytest
import numpy  as np
import pandas as pd
from   dataclasses import dataclass

from pandas      import DataFrame
from invisible_cities.io.mcinfo_io import load_mchits_df
from invisible_cities.io.mcinfo_io import load_mcparticles_df
from invisible_cities.io.mcinfo_io import load_mcsensor_response_df

from nextflex.mctrue_functions     import McParticles
from nextflex.mctrue_functions     import McHits
from nextflex.mctrue_functions     import get_mc_hits
from nextflex.reco_functions       import voxelize_hits
from nextflex.mctrue_functions     import get_event_hits_from_mchits
from nextflex.reco_functions       import get_voxels_as_list

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


@pytest.fixture(scope='session')
def bbonu_mc_particles(FDATA):
    testFile      = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Xe136_bb0nu.ACTIVE.0.next.h5')
    mcParticles  = load_mcparticles_df(testFile)
    return McParticles(mcParticles.sort_index())


@pytest.fixture(scope='session')
def bbonu_mc_hits(FDATA):
    testFile      = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Xe136_bb0nu.ACTIVE.0.next.h5')
    mcHits  = load_mchits_df(testFile)
    return McHits(mcHits.sort_index())


@pytest.fixture(scope='session')
def bbonu_hits_and_voxels(FDATA):
    testFile      = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Xe136_bb0nu.ACTIVE.0.next.h5')
    mcHits    = get_mc_hits(testFile)
    eventHits = get_event_hits_from_mchits(mcHits, event_id = 0)
    voxHits   = voxelize_hits(eventHits, bin_size = 10, baryc = True)
    return eventHits, voxHits


@pytest.fixture(scope='session')
def bbonu_and_1e_mchits(FDATA):
    testFilebb      = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Xe136_bb0nu.ACTIVE.0.next.h5')
    testFile1e      = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.e-.ACTIVE.1000.next.h5')

    mcHits_bb = get_mc_hits(testFilebb)
    mcHits_1e = get_mc_hits(testFile1e)
    return mcHits_bb, mcHits_1e


@pytest.fixture(scope='session')
def voxel_list(bbonu_hits_and_voxels):

    _, voxelHits = bbonu_hits_and_voxels
    return get_voxels_as_list(voxelHits)
