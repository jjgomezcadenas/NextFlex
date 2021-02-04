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

from nextflex.reco_functions import get_sipm_response
from nextflex.reco_functions import get_sipm_positions
from nextflex.reco_functions import get_event_sipm
from nextflex.reco_functions import get_sipm_event_hits

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
from nextflex.io import write_event_gtracks_json
from nextflex.io import load_event_gtracks_json
from nextflex.reco_functions import GTrack


def test_get_sipm_response(bb0nu_test_file):
    sipm_response = get_sipm_response(bb0nu_test_file)
    sipmr = sipm_response.df
    sipm_evts = get_index_slice_from_multi_index(sipmr, i = 0)
    assert np.allclose(sipm_evts, np.array([212, 213, 214, 215]))

    sipm_ids = get_index_slice_from_multi_index(sipmr, i = 1)
    assert len(sipm_ids) == 3093  # nof sipm in FLEX, this file


def test_get_sipm_positions(bb0nu_test_file) :
    sipm_positions = get_sipm_positions(bb0nu_test_file)
    sipmp = sipm_positions.df
    assert sipmp.sensor_id.min() == 1000
    assert sipmp.sensor_id.max() == 4092
    assert np.allclose(sipmp.x.min(), -482.05)
    assert np.allclose(sipmp.x.max(), 482.05)


def test_get_event_sipm(bb0nu_test_file):
    sipm_response = get_sipm_response(bb0nu_test_file)
    event_ids     = get_index_slice_from_multi_index(sipm_response.df, i = 0)
    sipm_evt      = get_event_sipm(sipm_response, event_ids[0])
    assert np.unique(sipm_evt.df.index.get_level_values("event_id"))[0] ==\
           sipm_evt.event_id

    assert len(get_index_slice_from_multi_index(sipm_evt.df, i = 1)) == 3093


def test_get_sipm_event_hits(bb0nu_test_file):
    sipm_positions = get_sipm_positions(bb0nu_test_file)
    sipm_response = get_sipm_response(bb0nu_test_file)
    event_ids     = get_index_slice_from_multi_index(sipm_response.df, i = 0)
    sipm_evt      = get_event_sipm(sipm_response, event_ids[0])
    sipm_hits     = get_sipm_event_hits(sipm_evt, sipm_positions, ecut = 10)

    assert sipm_hits.df.energy.max() == 292


def test_voxelize_hits(bbonu_hits_and_voxels):
    eventHits, voxelHits = bbonu_hits_and_voxels
    t12    = eventHits.df
    vt12df = voxelHits.df
    vHitsBar = voxelize_hits(eventHits, bin_size = 10, baryc = False)
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


def test_voxel_distances(bbonu_hits_and_voxels):
    _, voxelHits = bbonu_hits_and_voxels
    voxels = get_voxels_as_list(voxelHits)
    minimum_d, inclusive_d = voxel_distances(voxelHits)
    _, dist = np.histogram(minimum_d)
    assert dist[-1] < 10 # in mm
    _, dist = np.histogram(inclusive_d)
    assert dist[-1] < 100 # in mm

    vdp = voxel_distance_pairs(voxels) # must yields same results inclusive
    _, dist2 = np.histogram(vdp)
    assert np.allclose(dist, dist2)


def test_make_track_graphs(bbonu_hits_and_voxels, voxel_list):
    _, voxelHits = bbonu_hits_and_voxels
    vt12df       = voxelHits.df
    voxels       = voxel_list
    contiguity   = 20
    gtracks      = make_track_graphs(voxelHits, contiguity)
    gtvdf        = gtrack_voxels(gtracks[0])
    assert np.allclose(vt12df.x.values, gtvdf.x.values)
    assert np.allclose(vt12df.energy.values, gtvdf.energy.values)


def test_write_load_gtracks(bbonu_hits_and_voxels, voxel_list, FDATA):
    testFile     = os.path.join(FDATA,"testData",
                            'gtracks.json')
    eventGTracks = []
    for event_id in range(10):
        _, voxelHits = bbonu_hits_and_voxels
        #vt12df       = voxelHits.df
        #voxels       = voxel_list
        contiguity   = 20
        voxel_bin    = 10
        gtracks      = make_track_graphs(voxelHits, contiguity)
        GTRKS        = [GTrack(gtr, i, voxel_bin, contiguity)\
                        for i, gtr in enumerate(gtracks)]
        eventGTracks.append(GTRKS)

    write_event_gtracks_json(eventGTracks, testFile)
    eventGtraksFromFile  = load_event_gtracks_json(testFile)

    for i, gTracks in enumerate(eventGtraksFromFile):
        egTracks = eventGTracks[i] # tracks in the list of events
        egtrack = egTracks[0] # first track
        gtrack  = gTracks[0]
    assert egtrack.event_id == gtrack.event_id
    assert egtrack.voxel_bin == gtrack.voxel_bin
    assert egtrack.contiguity == gtrack.contiguity
