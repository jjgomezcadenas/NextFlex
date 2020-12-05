
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

if __name__ == "__main__":

    iPATH   = "/Users/jj/Development/flexData/"
    VERBOSITY = True

    setup = Setup(name = 'FLEX100_M6_O6_thr_0_nomem_el1bar',
                  sipmPDE    = 0.4,
                  maskPDE    = 1.0,  # no membrane
                  qTh        = 0.0,
                  maskConfig = "FLEX100_M6_O6", # height 6 mm hole 6 mm
                  mapDIR     = "flexmaps")

    ifnames = glob.glob(f"{iPATH}/{setup.maskConfig}/*.h5")

    if VERBOSITY:
        print(f"{len(ifnames)} input file names ...\n")
        for ifname in ifnames:
            print(ifname)
        #print(get_event_numbers_in_file(ifname))

    sipm_map = pd.read_csv(f'{iPATH}/{setup.mapDIR}/sipm_map.csv')
    print(sipm_map)

    gf, bf = kr_dst(ifnames, sipm_map)
    print(f'good files ={gf}')
    print(f'bad files ={bf}')

    ifnames = glob.glob(f"{iPATH}/{setup.maskConfig}/*.csv")
    ofile = f"{iPATH}/krdst_{setup.name}.csv"

    krdf, BF = kr_join_dst(ifnames, verbose=False)
    print(f'bad files ={BF}')
    print(krdf)
    krdf.to_csv(ofile)
