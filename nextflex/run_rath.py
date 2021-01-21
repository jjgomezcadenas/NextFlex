"""
- Run the reconstruction analysis for True hits (RATH).
    - The RATH voxelises MC True Hits, then computes TrackGraphs
    and find voxels extremes.
    - The driver returns an object containing relevant
    statistics and a EventTrack object, that is, a list,
    indexed by event number of GTRacks.
- Saves to file

#### The driver


@dataclass
class RecoGtrackFromMcHits:


    This class collects all information relevant for the topological
    analysis of reconstructed tracks (from monte carlo hits), plus
    statistics information.

    recoSetup           : Setup
    voxel_bin           : float
    contiguity          : float
    gtracks             : List[GTracks]        = None
    tExtrema            : pd.DataFrame         = None
    trackRecoStats      : pd.DataFrame         = None
    trackRecoTiming     : pd.DataFrame         = None
    tVoxelizationXYZ    : pd.DataFrame         = None
    trackRecoEventStats : TrackRecoEventStats  = None


- The driver for the reconstruction of gtracks (graph tracks) from Monte Carlo
hits takes as parameters:
    - The setup which is used internally for bookeeping
      (where to read/write files)
    - The voxel_bin and contiguity which define the voxelisation
    - The topology specifying which hits are used from the MC Hits
      (all or only from primary particles)
    - The event type (which can be bb0nu or 1e)
    - The baryc parameter which specifies if the position of the voxel is
      computed from the barycenter of the hits or from the average
    - debug, file_range to use and frequency of printing

- The driver returns an object **RecoGtrackFromMcHits** which has the following
     fields:
    - Setup, voxel_bin and contiguity.
    - a list of GTracks (each GTracks object is a list of GTrack or
       graph-track object)
    - a DF describing the true extrema for each track
    - Track reconstruction statistics (a DF)
    - Track reconstruction timing (a DF)
    - a DF which allows measurement of voxelisation timing versus
      number of voxels
    - Information on track reconstruction event statistics

"""
import os
import sys
import glob
import time
import warnings
import logging

import numpy  as np
import pandas as pd


from nextflex.core import Setup
from nextflex.reco_analysis import reco_gtrack_from_mc_hits


if __name__ == "__main__":


    event_types = ["bb0nu", "1e"]
    files = {"bb0nu" : "FLEX100_M6_O6.EL8bar.bb0nu",
             "1e"    : "FLEX100_M6_O6.EL8bar.1e"}

    # path to data directories
    FDATA = os.environ['FLEXDATA']
    print(f'path to data directories ={FDATA}')

    # define setup
    setup = Setup(flexDATA = FDATA,
                  tpConfig = files["bb0nu"])

    print(setup)

    voxel_bin  = 2
    contiguity = 10

    rgt  = reco_gtrack_from_mc_hits(setup, voxel_bin, contiguity,
                                topology   = "all",
                                event_type = "bb0nu",
                                baryc      = True,
                                debug      = False,
                                file_range = (0, -1),
                                ic         = 25)

    eff = rgt.trackRecoEventStats.e_gt/rgt.trackRecoEventStats.e_total

    print(f"1 track selection efficiency = {eff}")
    rgt.write_setup()
