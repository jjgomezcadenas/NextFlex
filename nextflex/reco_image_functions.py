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
from invisible_cities.io.dst_io    import load_dst

from  tics.system_of_units import *
from  tics.stats_tics import bin_data_with_equal_bin_size
from  tics.pd_tics    import get_index_slice_from_multi_index
from tics.util_tics   import find_nearest

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


@dataclass
class Point:
    x : float
    y : float
    z : float


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
    #fdf = fdf.drop(columns=['z']).reset_index(drop=True)
    #print(fdf)
    ix = fdf.x.max()
    iy = fdf.y.max()

    #print(ix, iy)
    xl = np.zeros((ix+1,iy+1))
    xe = np.zeros((ix+1,iy+1))

    #print(xl, xe)

    for i in fdf.index:
        xye = fdf.loc[slice(i,i),:].values[0]

        #print(xye)
        x = int(xye[0])
        y = int(xye[1])
        e = xye[3]

        #print(x, y, e)

        xl[x][y] = 1
        xe[x][y] = e

    lbl = label(xl,connectivity=2)
    prp = regionprops(lbl, xe)
    return SckLabelProp(fdf, xl, xe, lbl, prp)


def select_slice_by_label(dfz      : pd.DataFrame,
                          sckl     : SckLabelProp,
                          min_size : int = 1)->pd.DataFrame:
    """
    Select dfz in terms of the labeling specified by lbl
    Selects all labels larger than min_size

    """
    X = []
    Y = []
    E = []

    # selected region must have a size larger than min_size
    # (setting min_size = 1, kills isolated SiPMs)
    cond_size = np.array([1 \
                       if len(sckl.prp[lbl -1]['intensity_image']) >= min_size \
                       else 0 \
                       for lbl in sckl.lbl.flatten()]).reshape(*sckl.lbl.shape)

    #print(f" xe = {sckl.xe}\n")
    #print(f" lbl = {sckl.lbl}\n")
    #print(f" cond_size = {cond_size}\n")
    # select the indexes that veryfy condition above
    #c1 = np.argwhere(cond_size)
    #c2 = np.argwhere(sckl.lbl > 0)
    #c = c1 & c2
    #print(c1)
    #print(c2)
    #print(c)
    xlp = np.argwhere((cond_size) & (sckl.lbl > 0))

    if len(xlp) == 0:
        return None

    #print(f" index selected = {xlp}\n")
    #print(f" input data frame ={dfz}\n")
    #print(f" norm data frame ={sckl.fdf} \n")
    for el in xlp:
        #print(f" xl = {el[0]}, yl = {el[1]}")
        fs = sckl.fdf[(sckl.fdf.x == el[0]) & (sckl.fdf.y == el[1])]
        #print(fs.energy.values[0])
        dfs = dfz[dfz.energy == fs.energy.values[0]]
        X.append(dfs.x.values[0])
        Y.append(dfs.y.values[0])
        E.append(dfs.energy.values[0])
    df = pd.DataFrame({'x' : X, 'y' : Y, 'energy' : E})

    #print(f"selected df = {df}")
    return df


def detector_grid(dfzs : pd.DataFrame, bin_size : Tuple[float, float]):
    """
    Returns the grid needed for interpolation

    """
    #print(f" in detector_grid : dfzs = {dfzs}")
    #print(f" in detector_grid : bin_size = {bin_size}")
    return [np.arange(dfzs[var].min() + bs/2,
                      dfzs[var].max() - bs/2 + np.finfo(np.float32).eps, bs) \
          for var, bs in zip(['x', 'y'], bin_size)]


def image_from_df(df           : pd.DataFrame,
                  det_grid     : np.array,
                  sample_width : Tuple[float, float]= (15.5, 15.5))->np.array:
    """
    Compute an image suitable for imshow from the DF

    """
    #print(f"detector_grid = {det_grid}")
    data = (df.x.values, df.y.values)
    #print(f"data = {data}\n")
    weight = df.energy.values
    #print(f"weight = {weight}\n")
    ranges = [[coord.min() - 1.5 * sw, coord.max() + 1.5 * sw] \
              for coord, sw in zip(data, sample_width)]

    #print(f"ranges = {ranges}\n")
    allbins = [grid[in_range(grid, *rang)] \
               for rang, grid in zip(ranges, det_grid)]
    #print(f"allbins = {allbins}\n")
    hs, edges = np.histogramdd(data, bins=allbins, normed=False, weights=weight)
    return hs


def get_psf(pspath :str, psfname :str, dfzs : pd.DataFrame, xyz : Point):
    """
    Returns the PSF needed for deconvolution

    """
    psfile = os.path.join(pspath,psfname)
    psfs          = load_dst(psfile, 'PSF', 'PSFs')
    psf = psfs.loc[(psfs.z == find_nearest(psfs.z, xyz.z)) &
               (psfs.x == find_nearest(psfs.x, xyz.x)) &
               (psfs.y == find_nearest(psfs.y, xyz.y)) , :]
    var_name     = np.array(['xr', 'yr', 'zr'])
    data = (dfzs.x.values, dfzs.y.values)
    columns       = var_name[:len(data)]
    psf_deco      = psf.factor.values.reshape(psf.loc[:, columns].nunique().values)
    return psf_deco


def print_region_properties(regions):
    print(f" number of regions = {len(regions)}")
    for i, r in enumerate(regions):
        print(f"""
        region      = {i}
        label       = {r['label']}
        area        = {r['area']}
        coordinates = {r['coords']}
        intensiy    = {r['intensity_image']}
        """)
