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
from invisible_cities.io.mcinfo_io import load_mcconfiguration
from invisible_cities.io.mcinfo_io import load_mcsensor_positions

from nextflex.core import NN
from nextflex.core import SIPM_ids
from nextflex.core import Setup

def find_pitch(fileName : str)->float:
    """
    Find the pitch in the configuration of the MC file and returns it

    """
    def purge_list(lst):
        return list(dict.fromkeys(lst))

    mcConfig = load_mcconfiguration(fileName)
    mcConfig.set_index("param_key", inplace = True)
    par = mcConfig.loc["/Geometry/NextFlex/tp_sipm_pitchX"]
    #print(par)
    vals = purge_list(par.param_value.split(' '))
    #print(float(vals[1]))
    return float(vals[1])


def sipm_indx(sipm_map, eps=0.1):
    """
    Takes a dataframe with the sipm_map and adds de sipm indexes of xl, xr, yl, yr

    """
    def index_search(xv, yv, XL):
        spx  = sipm_map[in_range(sipm_map.x, xv - eps, xv + eps)]
        spxy = spx[in_range(spx.y, yv - eps, yv + eps)]
        if len(spxy.sensor_id.values) == 1:
            XL.append(spxy.sensor_id.values[0])
        else:
            XL.append(NN)
    iXL=[]
    iXR=[]
    iYU=[]
    iYD=[]
    for i in sipm_map.index:
        sipm = sipm_map.loc[i]
        index_search(sipm.xl, sipm.y,  iXL)
        index_search(sipm.xr, sipm.y,  iXR)
        index_search(sipm.x,  sipm.yu, iYU)
        index_search(sipm.x,  sipm.yd, iYD)

    sipm_map['id_xl'] = iXL
    sipm_map['id_xr'] = iXR
    sipm_map['id_yu'] = iYU
    sipm_map['id_yd'] = iYD


def sipm_pos(sipm_positions : DataFrame,
             pitch          : float,
             eps            : float = 0.1):
    """
    Takes a dataframe with the sipm_positions and:
    1. drops z position (not needed)
    2. for each sipm computes the xl, xr, yu, yd positions corresponding to:
       xl : sipm to the left along x axis of the sipm at position (x,y)
       xr : sipm to the right along x axis of the sipm at position (x,y)
       yu : sipm up along y axis of the sipm at position (x,y)
       yd : sipm down along y axis of the sipm at position (x,y)

    """

    def return_index(x,y):
        spx = sipm_positions[in_range(sipm_positions.x, x-eps, x+eps)]
        spxy = spx[in_range(spx.y, y-eps, y+eps)]
        if len(spxy.sensor_id.values) > 0:
            return spxy.sensor_id.values[0]
        else:
            return NN


    def compute_position(px, py, PMIN, PMAX, coord):

        xl, xr = px - pitch, px + pitch
        if coord == 'x':
            ixl = return_index(xl,py)
            ixr = return_index(xr,py)
        else:
            ixl = return_index(py,xl)
            ixr = return_index(py,xr)

        if ixl   == NN:
            xl = NN
        elif ixr == NN:
            xr = NN

        return xl, xr

    XMAX = sipm_positions.x.max()
    XMIN = sipm_positions.x.min()
    YMAX = sipm_positions.y.max()
    YMIN = sipm_positions.y.min()

    xL = []
    xR = []
    yU = []
    yD = []

    X = sipm_positions.x.values
    Y = sipm_positions.y.values

    new_positions = (sipm_positions.drop(columns=['z','index'])).copy()
    for xpos, ypos in zip(X,Y):

        xl, xr = compute_position(xpos, ypos, XMIN, XMAX, coord='x')
        yu, yd = compute_position(ypos, xpos, YMIN, YMAX, coord='y')


        xL.append(xl)
        xR.append(xr)
        yU.append(yu)
        yD.append(yd)

    new_positions['xl'] = xL
    new_positions['xr'] = xR
    new_positions['yu'] = yD  # yD is really yup
    new_positions['yd'] = yU

    return new_positions


if __name__ == "__main__":

    FDATA = os.environ['FLEXDATA']
    print(f'path to data directories ={FDATA}')

    # define setup
    setup = Setup(sipmPDE    = 1.0,
                  maskPDE    = 1.0,
                  qTh        = 0.0,
                  tpConfig   = "FLEX100_M6_O6_P10.EL8bar")
    print(setup)


    iPATH =  f"{FDATA}/{setup.tpConfig}"
    ifname = glob.glob(f"{iPATH}/*.h5")[0]
    pitch = find_pitch(ifname)
    print(f'for file {ifname} :  pitch ={pitch : 2.1f} mm ')

    mapPath = f"{FDATA}/flexmaps"
    mapName = f"{mapPath}/sipm_map_{pitch:2.1f}_mm.csv"
    print(f" map name = {mapName}")

    sns_positions = load_mcsensor_positions(ifname)
    print(sns_positions.head())
    sipm_positions = sns_positions[in_range(sns_positions.sensor_id,
    *SIPM_ids)].reset_index()

    print(sipm_positions.head())

    XMAX = sipm_positions.x.max()
    XMIN = sipm_positions.x.min()
    YMAX = sipm_positions.y.max()
    YMIN = sipm_positions.y.min()
    print(f'XMAX = {XMAX} XMIN = {XMIN} YMAX = {YMAX} YMIN = {YMIN}')

    sipm_map = sipm_pos(sipm_positions, pitch)
    sipm_indx(sipm_map)

    print(sipm_map.head())

    sipm_map.to_csv(mapName)
