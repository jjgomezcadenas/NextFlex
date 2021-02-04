import numpy as np
import pandas as pd
import networkx as nx
import json
import os
from dataclasses import dataclass
from dataclasses import field
from dataclasses import asdict

from itertools   import combinations

from pandas      import DataFrame
from typing      import List, Tuple, Dict
from typing      import TypeVar

from invisible_cities.core.core_functions     import in_range

from  tics.system_of_units import *
from  tics.stats_tics import bin_data_with_equal_bin_size
from  tics.pd_tics    import get_index_slice_from_multi_index

from nextflex.types import  EventSiPM
from nextflex.types import  EventHits
from nextflex.types import  VoxelHits

from scipy.signal import convolve2d as conv2
from skimage import color, data, restoration
from skimage.measure import label, regionprops
from skimage import data, util
from skimage.measure import label
from skimage.measure._regionprops import RegionProperties

@dataclass
class SckLabelProp:
    fdf : pd.DataFrame
    xl  : np.array
    xe  : np.array
    lbl : np.array
    prp : RegionProperties


def label_slice(dfz : pd.DataFrame, pitch : float = 15.5)->SckLabelProp:
    """
    Perform the labeling of a data frame (dfz)
    representing a slice in z of a hit collection.

    1. Compute indexes from coordinates (function shift_values)
    2. drop z column (all same values)

    The resulting dataframe (fdf) has indexes in x and y which allows
    the definition of a contiguity matrix (xl) and an intensity matrix (xe).

    1. The contiguity matrix has "0" in those indexes corresponding to an empty SiPM
       and 1 to those in which we find a SiPM with charge
    2. The intensity matrix has "0" for empty SiPM and the value of SiPM charge otherwise.

    The contiguity matrix is used to computed a scikit-image label matrix (lbl).
    https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label

    The intensity matrix is used to compute a scikit region props matrix (prp).
    https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops

    The function return an object SckLabelProp collecting all the above info


    """
    def shift_values(df : pd.DataFrame,pitch)->pd.Series:
        """
        x -> x-xmin / pitch
        """
        d = {}
        xmin = df.min()
        xs = (df.values - xmin) / pitch
        return xs.astype(int)

    fdf = dfz.copy()
    fdf['x'] = fdf[['x']].apply(shift_values, args=(pitch,))
    fdf['y'] = fdf[['y']].apply(shift_values, args=(pitch,))
    fdf = fdf.drop(columns=['z']).reset_index(drop=True)

    ix = fdf.x.max()
    iy = fdf.y.max()

    xl = np.zeros((ix+1,iy+1))
    xe = np.zeros((ix+1,iy+1))

    for i in fdf.index:
        xye = fdf.loc[slice(i,i),:].values[0]
        x = xye[0]
        y = xye[1]
        e = xye[2]

        xl[x][y] = 1
        xe[x][y] = e

    lbl = label(xl,connectivity=2)
    prp = regionprops(lbl, xe)
    return SckLabelProp(fdf, xl, xe, lbl, prp)


def select_slice_by_label(dfz : pd.DataFrame,
                          sckl : SckLabelProp)->pd.DataFrame:
    """
    Select dfz in terms of the labeling specified by lbl

    """
    X = []
    Y = []
    E = []
    xlp = np.argwhere(sckl.lbl > 0)
    for el in xlp:
        #print(el[0], el[1])
        fs = sckl.fdf[(sckl.fdf.x == el[0]) & (sckl.fdf.y == el[1])]
        #print(fs.energy.values[0])
        dfs = dfz[dfz.energy == fs.energy.values[0]]
        X.append(dfs.x.values[0])
        Y.append(dfs.y.values[0])
        E.append(dfs.energy.values[0])
    return pd.DataFrame({'x' : X, 'y' : Y, 'energy' : E})
