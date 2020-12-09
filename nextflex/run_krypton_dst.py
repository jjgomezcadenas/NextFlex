
import os
import sys
import glob
import time
import warnings
import logging

import numpy  as np
import pandas as pd


from nextflex.krypton_dst import Setup
from nextflex.krypton_dst import kr_dst
from nextflex.krypton_dst import kr_join_dst

def clean_csv(ddir):
    os.chdir(ddir)
    print(f'now in directory {os.getcwd()}')
    print(f'removing .csv files')
    os.system('rm *.csv')


if __name__ == "__main__":

    # path to data directories
    FDATA = os.environ['FLEXDATA']
    print(f'path to data directories ={FDATA}')

    # define setup
    setup = Setup(sipmPDE    = 0.4,
              maskPDE    = 0.4,
              qTh        = 5.0,
              tpConfig   = "FLEX100_M6_O6_EL8bar_memb")
    print(setup)

    # clean csv files
    ddir = f"{FDATA}/{setup.tpConfig}"
    clean_csv(ddir)

    # collect .h5 files
    ifnames = glob.glob(f"{ddir}/*.h5")
    print(f'found {len(ifnames)} files')

    # sipm map
    mapFile       = os.path.join(FDATA,setup.mapDIR, 'sipm_map.csv')
    sipm_map      = pd.read_csv(mapFile)

    # krdsts (csv files)
    gf, bf = kr_dst(ifnames, sipm_map, setup, ic=10)
    print(f'good files ={len(gf)}')
    print(f'bad files ={len(bf)}')

    # collect csv files
    ifnames2 = glob.glob(f"{FDATA}/{setup.tpConfig}/*.csv")
    print(f'found {len(ifnames2)} files')
    ofile_name = f"{setup.name}.csv"
    ofile = f"{FDATA}/kdsts/{ofile_name}"
    print(f"Path to ofile = {ofile}")
    krdf, BF = kr_join_dst(ifnames2, verbose=False)
    print(f'bad files ={len(BF)}')
    #print(krdf)
    krdf.to_csv(ofile)
