
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
from nextflex.krypton_dst import prepare_tmpdir
from nextflex.krypton_dst import collect_h5files
from nextflex.krypton_dst import collect_csvfiles
from nextflex.krypton_dst import get_file_name


if __name__ == "__main__":

    # path to data directories
    FDATA = os.environ['FLEXDATA']
    print(f'path to data directories ={FDATA}')

    # define setup
    setup = Setup(sipmPDE    = 1.0,
                  maskPDE    = 1.0,
                  qTh        = 0.0,
                  tpConfig   = "FLEX100_D3_M2_O6_EL8bar")
    print(setup)
    tmpdir = prepare_tmpdir(FDATA, setup)

    # collect .h5 files
    ifnames = collect_h5files(FDATA, setup)
    print(f'found {len(ifnames)} files')

    # sipm map
    mapFile       = os.path.join(FDATA,setup.mapDIR, 'sipm_map.csv')
    sipm_map      = pd.read_csv(mapFile)

    # krdsts (csv files)
    gf, bf = kr_dst(ifnames, FDATA, sipm_map, setup, ic=10)
    print(f'good files ={len(gf)}')
    print(f'bad files ={len(bf)}')

    # collect csv files
    ifnames2 = collect_csvfiles(FDATA, setup)
    print(f'found {len(ifnames2)} files')

    # create joined dst
    ofile_name = f"{setup.name}.csv"
    ofile = f"{FDATA}/kdsts/{ofile_name}"
    print(f"Path to ofile = {ofile}")
    krdf, BF = kr_join_dst(ifnames2, verbose=False)
    print(f'bad files ={len(BF)}')
    #print(krdf)
    krdf.to_csv(ofile)
