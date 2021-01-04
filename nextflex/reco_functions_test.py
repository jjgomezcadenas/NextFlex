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


from nextflex.reco_functions import voxelize_hits
from nextflex.reco_functions import get_voxels_as_list
from nextflex.reco_functions import voxel_position
from nextflex.reco_functions import voxel_energy
from nextflex.reco_functions import voxel_nhits
from nextflex.reco_functions import voxel_distances
from nextflex.reco_functions import distance_between_two_voxels
from nextflex.reco_functions import voxel_distance_pairs
from nextflex.reco_functions import make_track_graphs
from nextflex.reco_functions import gtrack_voxels

def test_voxelize_hits(bbonu_hits_and_voxels):
    eventHits, voxelHits = bbonu_hits_and_voxels
    t12    = eventHits.df
    vt12df = voxelHits.df
    vHitsBar = voxelize_hits(eventHits, bin_size = 5, baryc = False)
    gt12df = vHitsBar.df
    _, edx = np.histogram(np.abs(gt12df.x.values - vt12df.x.values))
    _, edy = np.histogram(np.abs(gt12df.y.values - vt12df.y.values))
    _, edz = np.histogram(np.abs(gt12df.z.values - vt12df.z.values))

    assert edx[-1] < 2 # difference between baryc and mean < 2 mm
    assert edy[-1] < 2 # difference between baryc and mean < 2 mm
    assert edz[-1] < 2 # difference between baryc and mean < 2 mm
    assert np.allclose(gt12df.nhits.values, vt12df.nhits.values)


def test_voxels_as_list(bbonu_hits_and_voxels):

    _, voxelHits = bbonu_hits_and_voxels
    vt12df       = voxelHits.df
    vlist        = np.array(get_voxels_as_list(voxelHits))
    assert np.allclose(vlist, vt12df.values)


def test_voxel_distances(voxel_list):
    voxels = voxel_list
    minimum_d, inclusive_d = voxel_distances(voxels)
    _, dist = np.histogram(minimum_d)
    assert dist[-1] < 7 # in mm
    _, dist = np.histogram(inclusive_d)
    assert dist[-1] < 100 # in mm

    vdp = voxel_distance_pairs(voxels) # must yields same results inclusive
    _, dist2 = np.histogram(vdp)
    assert np.allclose(dist, dist2)


def test_make_track_graphs(bbonu_hits_and_voxels, voxel_list):
    _, voxelHits = bbonu_hits_and_voxels
    vt12df       = voxelHits.df
    voxels     = voxel_list
    contiguity = 10
    gtracks    = make_track_graphs(voxels, contiguity)
    gtv        = gtrack_voxels(gtracks[0], voxelHits.event_id)
    gtvdf      = gtv.df
    assert np.allclose(vt12df.x.values, gtvdf.x.values)
    assert np.allclose(vt12df.energy.values, gtvdf.energy.values)


def test_blobd(bbonu_hits_and_voxels):
    _, voxelHits = bbonu_hits_and_voxels
