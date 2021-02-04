"""
Tests for reco_functions
"""
import os
import pytest
import numpy  as np
import tables as tb
from pytest           import approx

import numpy          as np
import pandas as pd

from  tics.pd_tics    import get_index_slice_from_multi_index

from nextflex.reco_image_functions import label_slice
from nextflex.reco_image_functions import select_slice_by_label


def test_label_slice(sipm_hits):

    lbl_test = np.array([[0, 0, 1, 0, 0],
       [0, 0, 1, 1, 0],
       [0, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 0]])
    img_test = np.array([[  0.,   0.,  11.,   0.,   0.],
       [  0.,   0.,  13.,  17.,   0.],
       [  0.,  26.,  32.,  30.,  15.],
       [ 22.,  67.,  82.,  32.,  25.],
       [ 19.,  96., 193., 114.,  19.],
       [ 26., 117., 272., 149.,  13.],
       [ 25.,  60., 118.,  61.,  21.],
       [ 14.,  12.,  24.,  13.,   0.]])

    zsl = np.unique(sipm_hits.df.z.values)
    assert len(zsl) == 57

    len_slices = [len(sipm_hits.df[sipm_hits.df.z == zs]) for zs in zsl]
    lmax = np.max(len_slices)
    ilmax = np.argmax(len_slices)
    assert lmax == 31
    assert ilmax == 27

    dfzm = sipm_hits.df[sipm_hits.df.z == zsl[ilmax]]
    sckl = label_slice(dfzm)

    assert np.allclose(sckl.lbl, lbl_test)
    assert np.allclose(sckl.lbl, sckl.xl.astype(int))
    assert np.allclose(sckl.xe, img_test)
    assert np.allclose(sckl.xe, sckl.prp[0]['intensity_image'])
    assert np.allclose(sckl.fdf.energy.values, dfzm.energy.values)


def test_select_slice_by_label(sipm_hits) :
    zsl = np.unique(sipm_hits.df.z.values)
    len_slices = [len(sipm_hits.df[sipm_hits.df.z == zs]) for zs in zsl]
    ilmax = np.argmax(len_slices)

    dfzm = sipm_hits.df[sipm_hits.df.z == zsl[ilmax]]
    sckl = label_slice(dfzm)


    dfzs = select_slice_by_label(dfzm, sckl)
    e1 = np.sort(dfzm.energy.values)
    e2 = np.sort(dfzs.energy.values)
    assert np.allclose(e1, e2)
