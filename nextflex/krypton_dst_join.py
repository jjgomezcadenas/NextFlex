# krypton_dst.py

import os
import sys
import glob
import time
import warnings

import numpy  as np
import pandas as pd

# Specific IC stuff
import invisible_cities.core.system_of_units  as units


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

    VERBOSITY = False
    iPATH = "/Users/jj/Development/demoData/"
    iDIR  = "FLEX100_M6_O6_UV"
    ofile = f"{iPATH}/{iDIR}/krdst.csv"
    ifnames = glob.glob(f"{iPATH}/{iDIR}/FLEX100*.csv")

    print(f"{len(ifnames)} input file names ...\n")

    if VERBOSITY:
        for ifname in ifnames:
            print(ifname)
        #print(get_event_numbers_in_file(ifname))

    krdf, BF = kr_join_dst(ifnames, verbose=False)
    print(f'bad files ={BF}')
    print(krdf)
    krdf.to_csv(ofile)
