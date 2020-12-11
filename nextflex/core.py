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
from typing import List, Tuple
from typing import Union
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
from invisible_cities.io.mcinfo_io import load_mcsensor_response_df

NN= -999999  # No Number, a trick to aovid nans in data structs

KEY_sensor_fibres  = 100000
KEY_sensor_pmts    = 100
S1_time            = 1.*units.mus
SIPM_ids           = (KEY_sensor_pmts, KEY_sensor_fibres)


class NNN:
    def __getattr__(self, _):
        return NN


@dataclass
class Setup:
    sipmPDE           : float = 1.0
    maskPDE           : float = 1.0
    qTh               : float = 0.0
    tpConfig          : str   = "FLEX100_M6_O6" # means thickness 6 mm hole 6mm
    mapDIR            : str   = "flexmaps" # where to find the SiPM map
    fibres            : bool  = False

    def __post_init__(self):
        if self.fibres:
            ens = 'Fibres'
        else:
            ens ="PMTs"
        name      = f"{ens}_sipmPDE_{self.sipmPDE}"
        name      = f"{name}_maskPDE_{self.maskPDE}_qTh_{self.qTh}"
        self.name = f"{name}_{self.tpConfig}"

    def __repr__(self):
        s = f"""
        Setup <{self.name}>:
        tracking plane configuration = {self.tpConfig}
        sipm PDE                     = {self.sipmPDE}
        transmission of teflon masks = {self.maskPDE}
        charge threshold             = {self.qTh}
        Fibres?                      = {self.fibres}
        """
        return s

    def __str__(self):
        return self.__repr__()


@dataclass
class PosQ:
    evt_list  : List[int]
    def __post_init__(self):
        llen = len(self.evt_list)
        self.xMax = np.zeros(shape=llen, dtype=float)
        self.xPos = np.zeros(shape=llen, dtype=float)
        self.yMax = np.zeros(shape=llen, dtype=float)
        self.yPos = np.zeros(shape=llen, dtype=float)
        self.rPos = np.zeros(shape=llen, dtype=float)
        self.Qtot = np.zeros(shape=llen, dtype=float)
        self.qMax = np.zeros(shape=llen, dtype=float)
        self.qL   = np.zeros(shape=llen, dtype=float)
        self.qR   = np.zeros(shape=llen, dtype=float)
        self.qU   = np.zeros(shape=llen, dtype=float)
        self.qD   = np.zeros(shape=llen, dtype=float)
    def to_dict(self):
        return {
            'event_id' : self.evt_list,
            'xMax': self.xMax,
            'yMax': self.yMax,
            'xPos': self.xPos,
            'yPos': self.yPos,
            'rPos': self.rPos,
            'Qtot': self.Qtot,
            'qMax': self.qMax,
            'qL'  : self.qL,
            'qR'  : self.qR,
            'qU'  : self.qU,
            'qD'  : self.qD
        }


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


def get_sensor_response(sns_response : DataFrame,
                        sensor_type  : str = 'PMT')->DataFrame:
    """Returns a data frame with the charge and time of each sensor
    for each event:
        sensor_type = PMT, SiPM, Fibres

    """

    #sns_response = load_mcsensor_response_df(filename)

    if sensor_type == 'PMT':
        sensor_response = sns_response[sns_response.index.\
                get_level_values("sensor_id") <= KEY_sensor_pmts]
        return sensor_response
    elif sensor_type == 'FIBRES':
        sensor_response = sns_response[sns_response.index.\
                get_level_values("sensor_id") > KEY_sensor_fibres]
        return sensor_response
    else:
        sensor_response = sns_response[in_range(
        sns_response.index.get_level_values("sensor_id"), *SIPM_ids)]
        return sensor_response


def sensor_response_ti(sensor_response : DataFrame)->DataFrame:
    """ Takes a data frame representing a sensor reponse and integrates time"""

    grouped_multiple         = sensor_response.groupby(
                       ['event_id', 'sensor_id']).agg({'charge': ['sum']})
    grouped_multiple.columns = ['tot_charge']
    df                       = grouped_multiple.reset_index()
    return df


def event_sensor_response_ti(sensor_response : DataFrame,
                             event_id        : int)->DataFrame:
    """ Takes a data frame representing a sensor reponse,
    integrates time and return the slice corresponding to event_id

    """
    df = sensor_response_ti(sensor_response)
    return df[df.event_id == event_id]


def sensor_number_response_ti(sensor_response : DataFrame,
                              sensor_id       : int)->DataFrame:
    """ Takes a data frame representing a sensor reponse,
    integrates time and return the slice corresponding to sensor_id

    """
    df = sensor_response_ti(sensor_response)
    return df[df.sensor_id == sensor_id]


def mcparts_and_sensors_response(ifname : str,
                                 setup : Setup)->Union[Tuple[DataFrame],bool]:
    """Returns DataFrames with the responses of the sensors and the MCParts"""

    try:
        mcParts = load_mcparticles_df(ifname)
    except:
        print(f'Failed reading mcparticles ={ifname}')
        return False
    try:
        sns_response  = load_mcsensor_response_df(ifname)
    except:
        print(f'Failed reading sns_response ={ifname}')
        return False

    try:
        if setup.fibres:
            energy_sensors_response = get_sensor_response(sns_response,
                                      sensor_type = 'FIBRES')

        else: #PMTs
            energy_sensors_response = get_sensor_response(sns_response,
                                      sensor_type = 'PMT')

        #sns_response = load_mcsensor_response_df(ifname)
    except:
        print(f'Failed reading energy_sensors_response: file ={ifname}')
        return False
    try:
        sipm_response               = get_sensor_response(sns_response,
                                      sensor_type = 'SIPM')
    except:
        print(f'Failed reading sipm_response: file ={ifname}')
        return False

    return mcParts, energy_sensors_response, sipm_response


def get_s1(energy_sensors_response : DataFrame)->Series:
    """Add the energy before the 1st mus (S1)"""
    return energy_sensors_response[\
           energy_sensors_response.time < S1_time].\
           groupby('event_id').charge.sum()


def get_s2(energy_sensors_response : DataFrame)->Series:
    """Add the energy after the 1st mus (S2)"""
    return energy_sensors_response[\
           energy_sensors_response.time > S1_time].\
           groupby('event_id').charge.sum()

def get_Q(Q: np.array, setup : Setup)->np.array:
    """
    Compute the charge of SiPMs after fluctuations and
    threshold, for SiPMs in a given event

    Q is an array which contains the
    number of photons that hit the sensors (or the masks) for each
    sensor

    """

    # if the PDE of the SiPM is less than one we need to fluctuate
    # these numbers according to Poisson and then multiply by the PDE
    # same applies for the "PDE" of the masks
    if setup.sipmPDE != 1:
        Q = np.array([np.random.poisson(xi) for xi in Q]) * setup.sipmPDE

    if setup.maskPDE != 1:
        Q = np.array([np.random.poisson(xi) for xi in Q]) * setup.maskPDE

    # if there is a threshold we apply it now.
    if setup.qTh > 0:
        Q = np.array([q if q > setup.qTh else 0 for q in Q])

    return Q


def get_qtot(sipmdf: DataFrame, setup : Setup)->Series:
    """Compute the charge of SiPMs above a threshold"""
    return sipmdf[sipmdf.tot_charge > setup.qTh].\
                  groupby('event_id').tot_charge.sum()

def get_iq(sipmevt : DataFrame, ix : int)->int:
    """
    Return the position in array of the sipm with index
    ix in event sipmevt

    """
    if ix != NN:
        try:
            iq = sipmevt[sipmevt.sensor_id==ix].index[0]
        except:
            print(f'Warning no charge in SiPM with index ={ix}')
            iq = -1
    else:
        iq = -1
    return iq


def get_q(sipmevt : DataFrame, ix : int, setup : Setup)->float:
    """Return the charge of the sipm with index ix in event sipmevt"""
    if ix != NN:

        try:
            q = sipmevt[sipmevt.sensor_id==ix].tot_charge.values[0]
        except IndexError:
            print(f'Warning no charge in SiPM with index ={ix}')
            q = 0
    else:
        q = 0
    return q


def get_pos(vz : np.array, vq : np.array)->float:
    """Computes baricenter as the product of position vz and charge vq"""
    return np.dot(vz, vq) / np.sum(vq)


def get_position(event_list : List,
                 sipmdf     : DataFrame,
                 sipm_map   : DataFrame,
                 setup      : Setup)->DataFrame:
    """
    Computes the (x,y) position of the event:
    digital algorithm: - position of the SiPM with max charge
    analog  algorithm: baricenter using (l,r) and (u,p) of qmax

    """

    pq = PosQ(event_list)

    for ii, i in enumerate(event_list):

        evt  = sipmdf[sipmdf.event_id==i]
        Q    = get_Q(evt.tot_charge.values, setup)

        pq.Qtot[ii]  = Q.sum()
        qmax         = evt.tot_charge.max()
        iqmax        = evt[evt.tot_charge==qmax].sensor_id.values[0]
        pq.qMax[ii]  = Q.max()

        qmaxdf                   = sipm_map[sipm_map.sensor_id==iqmax]
        pq.xMax[ii], pq.yMax[ii] =  qmaxdf.x.values[0], qmaxdf.y.values[0]
        xl, xr                   =  qmaxdf.xl.values[0], qmaxdf.xr.values[0]
        yu, yd                   =  qmaxdf.yu.values[0], qmaxdf.yd.values[0]

        iqL = get_iq(evt, qmaxdf.id_xl.values[0])
        iqR = get_iq(evt, qmaxdf.id_xr.values[0])
        iqU = get_iq(evt, qmaxdf.id_yu.values[0])
        iqD = get_iq(evt, qmaxdf.id_yd.values[0])

        pq.qL[ii] = Q[iqL] if iqL > 0 else 0
        pq.qR[ii] = Q[iqR] if iqR > 0 else 0
        pq.qU[ii] = Q[iqU] if iqU > 0 else 0
        pq.qD[ii] = Q[iqD] if iqD > 0 else 0

        # if iqL < 0:
        #      pq.qL[ii] = 0
        # else:
        #      pq.qL[ii] = Q[iqL]
        # if iqR < 0:
        #     pq.qR[ii] = 0
        # else:
        #     pq.qR[ii] = Q[iqR]
        # if iqU < 0:
        #     pq.qU[ii] = 0
        # else:
        #      pq.qU[ii] = Q[iqU]
        # if iqD < 0:
        #      pq.qD[ii] = 0
        # else:
        #      pq.qD[ii] = Q[iqD]


        # pq.qL[ii] = get_q(evt, qmaxdf.id_xl.values[0], setup)
        # pq.qR[ii] = get_q(evt, qmaxdf.id_xr.values[0], setup)
        # pq.qR[ii] = get_q(evt, qmaxdf.id_yu.values[0], setup)
        # pq.qR[ii] = get_q(evt, qmaxdf.id_yd.values[0], setup)

        pq.xPos[ii] = get_pos(np.array([pq.xMax[ii], xl, xr]),
                             np.array([pq.qMax[ii], pq.qL[ii], pq.qR[ii]]))
        pq.yPos[ii] = get_pos(np.array([pq.yMax[ii], yu, yd]),
                             np.array([pq.qMax[ii], pq.qU[ii], pq.qD[ii]]))
        pq.rPos[ii] = np.sqrt(pq.yPos[ii]**2)+ np.sqrt(pq.xPos[ii]**2)

    return pd.DataFrame(pq.to_dict()).set_index('event_id')


def diff_pos(truedf : DataFrame, pqdf : DataFrame)->DataFrame:
    """
    Compute the difference between true and estimated positions
    truedf is the data frame with true positions
    posdf  is the data frame with estimated positions

    """
    DX = {'event_id' : truedf.index,
          'dxPos'    : truedf.true_x.values - pqdf.xPos.values,
          'dyPos'    : truedf.true_y.values - pqdf.yPos.values,
          'dxMax'    : truedf.true_x.values - pqdf.xMax.values,
          'dyMax'    : truedf.true_y.values - pqdf.yMax.values}
    dxdf = pd.DataFrame(DX)
    return dxdf.set_index('event_id')
