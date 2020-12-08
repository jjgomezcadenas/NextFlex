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
from nextflex.core import get_s1
from nextflex.core import get_s2
from nextflex.core import get_qtot
from nextflex.core import get_q
from nextflex.core import get_pos
from nextflex.core import get_position
from nextflex.core import diff_pos


def true_pos_(evt_truePos):
    """ Test checks that range of x, y, z positions is
    consistent with NEXT-100 diameter and length

    """
    assert np.all(in_range(evt_truePos.true_x, -500, 500))
    assert np.all(in_range(evt_truePos.true_y, -500, 500))
    assert np.all(in_range(evt_truePos.true_z, 0, 1200))


def pmt_response_(pmt_response):
    """Test checks that there 60 pmts and charge per time bin
    and pmt is smaller than 10 pes (QE included)
    """
    # pmt numbers
    assert np.all(in_range(pmt_response.index.get_level_values(1),
                  0, 60))
    # charge per time bin (includes QE)
    assert np.all(in_range(pmt_response.charge, 0, 10))


def sipm_response_(sipm_response):
    """Test chehcks indexes of SiPMs (4500 in total)
    and charge range per sipm and bin time (no PDE)
    """
    # index
    assert np.all(in_range(sipm_response.index.get_level_values(1),
                  1000,4500))
    # Q per time bin, no PDE
    assert np.all(in_range(sipm_response.charge, 0, 200))


def test_Setup():
    setup = Setup()
    assert setup.name              == 'NEXT100_PMTs_sipmPDE_1_maskPDE_1_qTh_0'
    assert setup.sipmPDE           == 1.0
    assert setup.maskPDE           == 1.0
    assert setup.qTh               == 0.
    assert setup.maskConfig        == "FLEX100_M6_O6"
    assert setup.mapDIR            == "flexmaps"
    assert setup.fibres            == False


def test_get_evt_true_positions_df(mc_sns_sipm_map):
    tsu    = mc_sns_sipm_map
    evt_truePos = get_evt_true_positions_df(tsu.mcParts)
    true_pos_(evt_truePos)


def test_get_evt_true_positions_and_energy(mc_sns_sipm_map):

    tsu               = mc_sns_sipm_map
    evt_truePos       = get_evt_true_positions_and_energy(tsu.mcParts)
    true_pos_(evt_truePos)
    assert np.all(in_range(evt_truePos.KE, 0, 60))


def test_get_sensor_response_pmts(mc_sns_sipm_map):
    """The values given in test correspond to the numbers of sensors"""

    tsu          = mc_sns_sipm_map
    pmt_response = get_sensor_response(tsu.sns_response_pmt,
                                       sensor_type = 'PMT')
    pmt_response_(pmt_response)


def test_get_sensor_response_fibres(mc_sns_sipm_map):
    """The values given in test correspond to the numbers of sensors"""
    tsu             = mc_sns_sipm_map
    fibres_response = get_sensor_response(tsu.sns_response_fib,
                                          sensor_type = 'FIBRES')
    assert np.all(in_range(fibres_response.index.get_level_values(1),
                           100000,201545))  # fibre numbers
    # charge per time bin (no QE)
    assert np.all(in_range(fibres_response.charge, 0, 200))


def test_get_sensor_response_sipms(mc_sns_sipm_map):
    """The values given in test correspond to the numbers of sensors"""
    tsu             = mc_sns_sipm_map
    sipm_response   = get_sensor_response(tsu.sns_response_pmt,
                                          sensor_type = 'SIPM')
    sipm_response_(sipm_response)


def test_sensor_response_ti_pmts(mc_sns_sipm_map):
    tsu          = mc_sns_sipm_map
    pmt_response = get_sensor_response(tsu.sns_response_pmt,
                                       sensor_type = 'PMT')
    pmtdf        = sensor_response_ti(pmt_response)
    assert np.all(in_range(pmtdf.sensor_id, 0, 60))
    assert np.all(in_range(pmtdf.tot_charge, 0, 300))


def test_sensor_response_ti_fibres(mc_sns_sipm_map):
    tsu             = mc_sns_sipm_map
    fibres_response = get_sensor_response(tsu.sns_response_fib,
                                          sensor_type = 'FIBRES')
    fibdf              = sensor_response_ti(fibres_response)
    assert np.all(in_range(fibdf.sensor_id, 100000,201545))

    fibdf2 = fibdf[fibdf.sensor_id>200000]
    assert np.all(in_range(fibdf2.tot_charge, 0, 40))


def test_sensor_response_ti_sipms(mc_sns_sipm_map):
    tsu             = mc_sns_sipm_map
    sipm_response   = get_sensor_response(tsu.sns_response_pmt,
                                          sensor_type = 'SIPM')
    sipmdf          = sensor_response_ti(sipm_response)
    assert np.all(in_range(sipmdf.sensor_id, 1000,4500))
    assert np.all(in_range(sipmdf.tot_charge, 0, 130))


def test_event_sensor_response_ti_pmts(mc_sns_sipm_map):
    tsu          = mc_sns_sipm_map
    pmt_response = get_sensor_response(tsu.sns_response_pmt,
                                       sensor_type = 'PMT')
    pmt_evt      = event_sensor_response_ti(pmt_response, event_id=100000)
    assert len(pmt_evt) == 60
    assert pmt_evt.tot_charge.sum() == 7752  # sum in pes


def test_event_sensor_response_ti_fibres(mc_sns_sipm_map):
    tsu             = mc_sns_sipm_map
    fibres_response = get_sensor_response(tsu.sns_response_fib,
                                          sensor_type = 'FIBRES')
    fibres_evt = event_sensor_response_ti(fibres_response, event_id=100)
    assert len(fibres_evt) == 3089
    assert fibres_evt.tot_charge.sum() == 50642


def test_event_sensor_response_ti_sipms(mc_sns_sipm_map):
    tsu             = mc_sns_sipm_map
    sipm_response   = get_sensor_response(tsu.sns_response_pmt,
                                          sensor_type = 'SIPM')
    sipm_evt        = event_sensor_response_ti(sipm_response, event_id=100000)
    assert sipm_evt.tot_charge.sum() == 795


def test_mcparts_and_sensors_response(FDATA):
    testFile  = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Kr83.ACTIVE.1000.next.h5')
    setup   = Setup()
    result  = mcparts_and_sensors_response(testFile, setup)
    mcParts, pmt_response, sipm_response = result

    evt_truePos = get_evt_true_positions_df(mcParts)
    true_pos_(evt_truePos)
    pmt_response_(pmt_response)
    sipm_response_(sipm_response)


def test_get_s1_s2(mc_sns_sipm_map):
    """Checks values of S1 and S2"""
    tsu          = mc_sns_sipm_map
    pmt_response = get_sensor_response(tsu.sns_response_pmt,
                                       sensor_type = 'PMT')
    s1           = get_s1(pmt_response)

    assert np.all(in_range(s1, 5, 45))  #range of S1 values for krypton

    s2 = get_s2(pmt_response)
    assert np.all(in_range(s2, 3000, 9000)) # Range of S2 values for krypton


def test_get_qtot(mc_sns_sipm_map):
    """Check value of total charge no cuts in the SiPM plane"""
    tsu             = mc_sns_sipm_map
    sipm_response   = get_sensor_response(tsu.sns_response_pmt,
                                          sensor_type = 'SIPM')
    sipmdf        = sensor_response_ti(sipm_response)
    setup         = Setup()
    qtot          = get_qtot(sipmdf, setup)

    assert np.all(in_range(qtot, 200, 900))  #range of qtot values for krypton


def test_get_q_and_get_pos(mc_sns_sipm_map):

    tsu           = mc_sns_sipm_map
    krdf          = get_evt_true_positions_and_energy(tsu.mcParts)
    sipm_response   = get_sensor_response(tsu.sns_response_pmt,
                                          sensor_type = 'SIPM')

    sipmdf        = sensor_response_ti(sipm_response)
    setup         = Setup()

    pqdf = get_position(krdf.index, sipmdf, tsu.sipm_map, setup)

    assert np.all(in_range(pqdf.xMax, -500, 500))
    assert np.all(in_range(pqdf.yMax, -500, 500))
    assert np.all(in_range(pqdf.xPos, -500, 500))
    assert np.all(in_range(pqdf.yPos, -500, 500))

    assert np.all(in_range(pqdf.qU, 0, 100))
    assert np.all(in_range(pqdf.qD, 0, 100))
    assert np.all(in_range(pqdf.qL, 0, 100))
    assert np.all(in_range(pqdf.qR, 0, 100))

    assert np.all(in_range(pqdf.rPos, 0, 700))
    assert np.all(in_range(pqdf.qMax, 0, 200))

def test_diff_pos(mc_sns_sipm_map):
    tsu           = mc_sns_sipm_map
    krdf          = get_evt_true_positions_and_energy(tsu.mcParts)
    sipm_response = get_sensor_response(tsu.sns_response_pmt,
                                          sensor_type = 'SIPM')
    sipmdf        = sensor_response_ti(sipm_response)
    setup         = Setup()
    pqdf = get_position(krdf.index, sipmdf, tsu.sipm_map, setup)

    dxdf = diff_pos(krdf, pqdf)

    qmax = 20
    assert np.all(in_range(dxdf.dxPos, -qmax, qmax))
    assert np.all(in_range(dxdf.dyPos, -qmax, qmax))
    assert np.all(in_range(dxdf.dxMax, -qmax, qmax))
    assert np.all(in_range(dxdf.dyMax, -qmax, qmax))
