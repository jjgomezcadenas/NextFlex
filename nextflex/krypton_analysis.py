import os
import sys
import glob
import time
import warnings
import logging

import numpy  as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


from pandas import DataFrame, Series
from typing import List

# Specific IC stuff
import invisible_cities.core.system_of_units  as units
from invisible_cities.core.core_functions     import in_range


def histo1d(var, varmin, varmax, xlabel, ylabel,
        bins=10, alpha=0.6, color='g'):

    plt.hist(var, bins=bins, alpha=alpha, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def histos1d(vars, varmins, varmaxs, xlabels, ylabels,
              bins, alphas, colors,
              splt=(1,2), figsize=(10,10)):

    fig = plt.figure(figsize=figsize)

    for i, var in enumerate(vars):
        ax = plt.subplot(*splt,i+1)
        histo1d(var, varmins[i], varmaxs[i], xlabels[i], ylabels[i],
        bins[i], alphas[i], colors[i])

    plt.tight_layout()
    plt.show()



def histo_df(df, var, varmin, varmax, xlabel, ylabel,
             bins=10, alpha=0.6, color='g'):

    df1 = df[in_range(df[var], varmin, varmax)]
    histo1d(df1[var].values, varmin, varmax, xlabel, ylabel,
        bins, alpha, color)


def histos_df(df, vars, varmins, varmaxs, xlabels, ylabels,
              bins, alphas, colors,
              splt=(1,2), figsize=(10,10)):

    fig = plt.figure(figsize=figsize)

    for i, var in enumerate(vars):
        ax = plt.subplot(*splt,i+1)
        histo_df(df, var, varmins[i], varmaxs[i], xlabels[i], ylabels[i],
        bins[i], alphas[i], colors[i])

    plt.tight_layout()
    plt.show()


def kr_point_resolution(krdst, xlim, bins=100, figsize=(10,10),
                        alpha=0.6, pitch=15):
    """Plots the resolution for Krypton: (xmax-xtrue), (xpos-xtrue),
    same for y

    """
    fig = plt.figure(figsize=figsize)

    varx =['xmax','xpos','ymax','ypos']
    vtru =['true_x','true_x','true_y','true_y']

    xmin = xlim[0]
    xmax = xlim[1]


    for i,var in enumerate(varx):
        ax      = fig.add_subplot(2, 2, i+1)
        vt = vtru[i]
        kfid = krdst[in_range(krdst[vt] - krdst[var], xmin, xmax)]
        dx = (kfid[vt] - kfid[var])
        hx = (krdst[vt] - krdst[var])

        if var == 'xpos' or var == 'ypos':
            mu, std = norm.fit(dx)
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
        else:
            std = pitch/np.sqrt(12)

        plt.hist(hx, bins=bins, density=True, range=(-20,20),
                 alpha=alpha, label=f'\u03C3 = {std:5.2f} mm')
        plt.xlabel(f'{var} (mm)')
        plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def kr_point_resolution2(krdst, xlim, bins=100, figsize=(10,10),
                        alpha=0.6, pitch=15):
    """Plots the resolution for Krypton: (xmax-xtrue), (xpos-xtrue),
    same for y

    """
    fig = plt.figure(figsize=figsize)

    varx =['dxMax', 'dyMax','dxPos', 'dyPos']

    xmin = xlim[0]
    xmax = xlim[1]


    for i,var in enumerate(varx):
        ax      = fig.add_subplot(2, 2, i+1)
        #print(var)
        kfid = krdst[in_range(krdst[var], xmin, xmax)]
        #print('dx')
        dx = kfid[var]
        #print('hx')
        hx = krdst[var]

        if var == 'dxPos' or var == 'dyPos':
            mu, std = norm.fit(dx)
            #print(mu, std)
            x = np.linspace(xmin, xmax, 100)
            #print(len(x))
            p = norm.pdf(x, mu, std)
            #print(len(p))
            plt.plot(x, p, 'k', linewidth=2)
        else:
            std = pitch/np.sqrt(12)

        plt.hist(hx, bins=bins, density=True, range=(-20,20),
                 alpha=alpha, label=f'\u03C3 = {std:5.2f} mm')
        plt.xlabel(f'{var} (mm)')
        plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def q_sipm(krdst,  bins=100, figsize=(10,10), alpha=0.6):
    """Plots the qmax, ql,qr,qu,qd"""

    Q = ['qL','qR','qU','qD']
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 2, 1)
    plt.hist(krdst.qMax, bins=bins, density=True, alpha=alpha, label=f'qmax')
    plt.legend()
    ax      = fig.add_subplot(1, 2, 2)
    for q in Q:
        plt.hist(krdst[q], bins=bins, density=True, alpha=alpha, label=f'{q}')
        plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def q4_sipm(krdst,  bins=100, figsize=(10,10), alpha=0.6):
    """Plots ql,qr,qu,qd"""

    Q = ['ql','qr','qu','qd']
    fig = plt.figure(figsize=figsize)

    for i, q in enumerate(Q):
        ax      = fig.add_subplot(2, 2, i+1)
        plt.hist(krdst[q], bins=bins, density=True, alpha=alpha, label=f'{q}')
        plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()
