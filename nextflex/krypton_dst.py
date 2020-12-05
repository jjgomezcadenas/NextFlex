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
from typing import List

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



def kr_dst(ifnames           : List[str],
           sipm_map          : DataFrame,
           setup             : Setup,
           verbose           : bool  = False,
           ic                : int   = 100):
    """Prepares an analysis dst for Krypton, including:

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

    def sipm_time_integral():
        grouped_multiple = sipm_response.groupby(
                           ['event_id', 'sensor_id']).agg({'charge': ['sum']})
        grouped_multiple.columns = ['tot_charge']
        return grouped_multiple.reset_index()

    def get_q(evt, ix):
        if ix != NN:

            try:
                q = evt[evt.sensor_id==ix].tot_charge.values[0]
            except IndexError:
                print(f'Warning no charge in SiPM adyacent to qmax, index ={ix}')
                q = 0
        else:
            q = 0
        return q * setup.sipmPDE * setup.maskPDE

    def get_pos(vz, vq):
        return np.dot(vz, vq) / np.sum(vq)

    def get_krdf():
        sipmdf = sipm_time_integral()

        if verbose:
            print(sipmdf)

        krdf = get_evt_true_positions_and_energy(mcParts)
        krdf['S1e'] = energy_sensors_response[\
                      energy_sensors_response.time < setup.s1_time].\
                      groupby('event_id').charge.sum()
        krdf['S2e'] = energy_sensors_response[\
                      energy_sensors_response.time > setup.s1_time].\
                      groupby('event_id').charge.sum()

        # krdf['Qtot'] = sipm_response.groupby('event_id').charge.sum()
        krdf['Qtot'] = sipmdf[sipmdf.tot_charge > setup.qTh].\
                       groupby('event_id').charge.sum()

        if verbose:
            print(krdf)

        xMax = []
        xPos = []
        yMax = []
        yPos = []
        rPos = []

        qMax = []
        qL   = []
        qR   = []
        qU   = []
        qD   = []

        if verbose:
            print(krdf.index)

        ii = 0
        for i in krdf.index:
            if ii%ic == 0:
                print(f' event = {ii} event number = {i}')

            ii+=1

            evt          = sipmdf[sipmdf.event_id==i]
            qmax         = evt.tot_charge.max()
            iqmax        = evt[evt.tot_charge==qmax].sensor_id.values[0]

            qmaxdf       = sipm_map[sipm_map.sensor_id==iqmax]
            xqmax, yqmax =  qmaxdf.x.values[0], qmaxdf.y.values[0]
            xl, xr       =  qmaxdf.xl.values[0], qmaxdf.xr.values[0]
            yu, yd       =  qmaxdf.yu.values[0], qmaxdf.yd.values[0]

            ql = get_q(evt, qmaxdf.id_xl.values[0])
            qr = get_q(evt, qmaxdf.id_xr.values[0])
            qu = get_q(evt, qmaxdf.id_yu.values[0])
            qd = get_q(evt, qmaxdf.id_yd.values[0])

            xp = get_pos(np.array([xqmax, xl, xr]), np.array([qmax, ql, qr]))
            yp = get_pos(np.array([yqmax, yu, yd]), np.array([qmax, qu, qd]))

            xMax.append(xqmax)
            xPos.append(xp)
            yMax.append(yqmax)
            yPos.append(yp)
            rPos.append(np.sqrt(yp**2)+ np.sqrt(xp**2))
            qMax.append(qmax)
            qL.append(ql)
            qR.append(qr)
            qU.append(qu)
            qD.append(qd)


        krdf['xmax'] = xMax
        krdf['ymax'] = yMax
        krdf['xpos'] = xPos
        krdf['ypos'] = yPos
        krdf['rpos'] = rPos
        krdf['qmax'] = qMax
        krdf['ql']   = qL
        krdf['qr']   = qR
        krdf['qu']   = qU
        krdf['qd']   = qD

        return krdf, ii

    # Glue files

    GF =[]
    BF =[]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for ifname in ifnames:
            print(f'reading file {ifname}')

            try:
                mcParts = load_mcparticles_df(ifname)
            except:
                print(f'Failed reading mcparticles ={ifname}')
                BF.append(ifname)
                continue

            try:
                sns_response = load_mcsensor_response_df(ifname)
            except:
                print(f'Failed reading sns_response ={ifname}')
                continue

            GF.append(ifname)
            sipm_ids =(setup.key_sensor_pmts, setup.key_sensor_fibres)

            if setup.fibres:
                energy_sensors_response = sns_response[\
                sns_response.index.\
                get_level_values("sensor_id") >= setup.key_sensor_fibres]
            else: #PMTs
                energy_sensors_response = sns_response[\
                sns_response.index.\
                get_level_values("sensor_id") <= setup.key_sensor_pmts]

            sipm_response = sns_response[\
                            in_range(sns_response.index.\
                            get_level_values("sensor_id"), *sipm_ids)]


            krdf, nof = get_krdf()
            file = get_file_name(ifname)
            print(f'saving file {file}, with {nof} events')
            krdf.to_csv(file)

    return GF, BF


def kr_join_dst(ifnames, verbose=False, ic=100):
    """Joins the csv dst files
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


if __name__ == "__main__":

    VERBOSITY = True

    setup = Setup(name = 'FLEX100_M6_O6_thr_5',
                  sipmPDE    = 0.4,
                  maskPDE    = 1.0,  # no membrane
                  qTh        = 5.0,
                  maskConfig = "FLEX100_M6_O6", # height 6 mm hole 6 mm
                  mapDIR     = "flexmaps")

    ifnames = glob.glob(f"{setup.iPATH}/{setup.maskConfig}/*.h5")

    if VERBOSITY:
        print(f"{len(ifnames)} input file names ...\n")
        for ifname in ifnames:
            print(ifname)
        #print(get_event_numbers_in_file(ifname))

    sipm_map = pd.read_csv(f'{setup.iPATH}/{setup.mapDIR}/sipm_map.csv')
    print(sipm_map)

    gf, bf = kr_dst(ifnames, sipm_map)
    print(f'good files ={gf}')
    print(f'bad files ={bf}')
