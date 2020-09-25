# krypton_dst.py

import os
import sys
import glob
import time
import warnings
import logging

import numpy  as np
import pandas as pd

# Specific IC stuff
import invisible_cities.core.system_of_units  as units

from invisible_cities.io.mcinfo_io import load_mcconfiguration
from invisible_cities.io.mcinfo_io import load_mcparticles_df
from invisible_cities.io.mcinfo_io import load_mchits_df
from invisible_cities.io.mcinfo_io import load_mcsensor_positions
from invisible_cities.io.mcinfo_io import load_mcsensor_response_df
from invisible_cities.io.mcinfo_io import get_sensor_types
from invisible_cities.io.mcinfo_io import get_sensor_binning
from invisible_cities.io.mcinfo_io import get_event_numbers_in_file
from invisible_cities.core.core_functions import in_range
from invisible_cities.core.core_functions import shift_to_bin_centers

from dataclasses import dataclass
from typing                import Tuple
from invisible_cities.core import fit_functions as fitf

def profile1d(z : np.array,
              e : np.array,
              nbins_z : int,
              range_z : np.array)->Tuple[float, float, float]:
    """Adds an extra layer to profileX, returning only valid points"""
    x, y, yu     = fitf.profileX(z, e, nbins_z, range_z)
    valid_points = ~np.isnan(yu)
    x    = x [valid_points]
    y    = y [valid_points]
    yu   = yu[valid_points]
    return x, y, yu


def closest_index(xa, xv):
    """Returns the index of the element in array xa closest to xv"""
    dx = np.abs(xa - xv)
    return dx.argmin()


@dataclass
class RMap:
    """histogram map"""
    R  : np.array
    e : np.array

    def map_i(self, r):
        """Return the ir coordinates in map corresponding to the location R"""
        return closest_index(self.R, r)

    def cr(self, r):
        """Return the correction in map corresponding to the location R"""
        i = self.map_i(r)
        ii = np.min([i, len(self.R) -1])
        return self.e[ii]


def rXY(x,y):
    return np.sqrt(x**2 + y**2)

def pXY(x,y):
    return np.arctan2(y,x)


def radial_energy_correction(krdst, rmap, varx='true_x', vary='true_y', verbose=False, ic=100):
    """Takes a radial map (rmap) and corrects for the dominant radial dependences"""
    ii = 0
    CC = np.zeros(len(krdst.index))
    EC = np.zeros(len(krdst.index))
    krdf = (krdst.drop(columns=['Unnamed: 0'])).copy()

    print(len(CC))
    #for i in range(10):
    for i in krdst.index:

        evt = krdst.loc[i]
        if len(evt) == 0:
            print(f' event = {ii} number = {i} is corrupted, skipping')
            continue

        if ii%ic == 0 and verbose:
            print(f' event = {ii} number = {i}, x = {evt[varx]}, y = {evt[vary]}, S2e = {evt.S2e}')

        r = rXY(evt.true_x,evt.true_y)
        c   = rmap.cr(r)
        CC[ii] = c
        if c > 0:
            EC[ii] = evt.S2e/c

        if ii%ic == 0 and verbose:
            print(f' c = {c}, Ec ={EC[ii]}')
        ii+=1

    krdf['Ec'] = EC
    krdf['gc'] = CC

    return krdf


def plot_ec(krdst, emin=0, emax=1e+6, bins=100, alpha=0.6, color='g'):

    kemax = krdst[krdst.Ec<emax]
    kemin = kemax[kemax.Ec>emin]
    ec    = kemin.Ec.values
    mu, std = norm.fit(ec)
    plt.hist(ec, bins=bins, density=True, alpha=alpha, color=color)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, bins)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    print(f' sigma/E ={2.3 * 100 * std/mu} FWHM')
    plt.show()
