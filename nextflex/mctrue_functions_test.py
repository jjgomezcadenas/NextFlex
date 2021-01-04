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


from tics.util_tics import get_class_name
from nextflex.mctrue_functions import get_mc_particles
from nextflex.mctrue_functions import select_mc_particles
from nextflex.mctrue_functions import get_mc_primary_particles
from nextflex.mctrue_functions import get_mc_hits
from nextflex.mctrue_functions import select_mc_hits
from nextflex.mctrue_functions import total_hit_energy
from nextflex.mctrue_functions import get_event_hits_from_mchits


def test_get_mc_particles(FDATA):
    testFile  = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Xe136_bb0nu.ACTIVE.0.next.h5')

    mcParticles = get_mc_particles(testFile)
    assert get_class_name(mcParticles) == "McParticles"
    assert mcParticles.columns == ('particle_name', 'primary', 'mother_id',
           'initial_x', 'initial_y', 'initial_z', 'initial_t',
           'final_x', 'final_y', 'final_z', 'final_t',
           'initial_volume', 'final_volume', 'initial_momentum_x',
           'initial_momentum_y', 'initial_momentum_z', 'final_momentum_x',
           'final_momentum_y', 'final_momentum_z', 'kin_energy', 'length',
           'creator_proc', 'final_proc')
    assert mcParticles.index == ('event_id', 'particle_id')
    assert np.allclose(mcParticles.event_list(),np.array([0, 1, 2, 3]))


def test_get_mc_primary_particles(bbonu_mc_particles):
    mcParticles = bbonu_mc_particles
    mcPrim = select_mc_particles(mcParticles,
                      event_slice=slice(None,None), particle_slice=slice(1,2))

    mc1 = mcPrim.df

    mcPrimary = get_mc_primary_particles(mcParticles)
    mc2 = mcPrimary.df

    for column in ('particle_name', 'primary', 'initial_volume',
                   'final_volume',
                   'creator_proc', 'final_proc'):
        np.equal(mc1[column].values, mc2[column].values)

    for column in ('mother_id',
           'initial_x', 'initial_y', 'initial_z', 'initial_t',
           'final_x', 'final_y', 'final_z', 'final_t',
            'initial_momentum_x',
           'initial_momentum_y', 'initial_momentum_z', 'final_momentum_x',
           'final_momentum_y', 'final_momentum_z', 'kin_energy', 'length'):
        np.allclose(mc1[column].values, mc2[column].values)


def test_select_mc_particles(bbonu_mc_particles):
    mcParticles = bbonu_mc_particles
    evt_ke = np.sum(select_mc_particles(mcParticles,
                    event_slice=slice(0,0),
                    particle_slice=slice(1,2),
                    columns='kin_energy'))

    assert np.allclose(evt_ke, 2.4578302)


def test_get_mc_hits(FDATA):
    testFile  = os.path.join(FDATA,"testData",
                            'FLEX100_M6_O6.Xe136_bb0nu.ACTIVE.0.next.h5')

    mcHits = get_mc_hits(testFile)
    assert get_class_name(mcHits) == "McHits"
    assert mcHits.columns == ('x', 'y', 'z', 'time', 'energy', 'label')
    assert mcHits.index == ('event_id', 'particle_id', 'hit_id')
    assert np.allclose(mcHits.event_list(),np.array([0, 1, 2, 3]))


def test_select_mc_hits(bbonu_mc_particles, bbonu_mc_hits):
    mcParticles = bbonu_mc_particles
    mcHits      = bbonu_mc_hits

    # Select hits for event = 0, all particles 1st hit
    # (as a trick to gather all particles)

    hits_00 = select_mc_hits(mcHits,
                        event_slice = slice(0,0),
                        particle_slice =slice(None, None),
                        hit_slice = slice(0, 0))

    # Compare with particles from McParticles
    parts_00 = select_mc_particles(mcParticles,
                                   event_slice=slice(0,0),
                                   particle_slice=slice(None,None))

    # particles must be the same in both objects
    vi = hits_00.df.index.values
    particle_id_1 = np.unique(list(zip(*vi))[1])
    vi = hits_00.df.index.values
    particle_id_2 = np.unique(list(zip(*vi))[1])

    assert np.allclose(particle_id_1, particle_id_2)


def test_total_hit_energy(bbonu_mc_hits):
    mcHits      = bbonu_mc_hits
    he00 = total_hit_energy(mcHits,
                            event_slice = slice(0,0),
                            particle_slice = slice(None, None))
    ev = he00.total_hit_energy.values[0]
    assert np.allclose(ev, 2.4578302)

    hep = total_hit_energy(mcHits, event_slice = slice(0,0),
                             particle_slice =slice(1, 2))
    ev1 = hep.total_hit_energy.values[0]
    assert np.allclose(ev1, 2.354522)

    hnp = total_hit_energy(mcHits, event_slice = slice(0,0),
                             particle_slice =slice(3, None))
    ev2 = hnp.total_hit_energy.values[0]

    assert np.allclose(ev2, 0.103308)
    assert np.allclose(ev, ev1 + ev2)


def test_get_event_hits_from_mchits(bbonu_mc_hits):
    mcHits      = bbonu_mc_hits
    mche = get_event_hits_from_mchits(mcHits, event_id=0, hit_type='primary')
    hep = total_hit_energy(mcHits, event_slice = slice(0,0),
                             particle_slice =slice(1, 2))
    ev1 = hep.total_hit_energy.values[0]
    mcthe = mche.df.energy.sum()
    assert np.allclose(ev1, mcthe)

    mcha = get_event_hits_from_mchits(mcHits, event_id=0, hit_type='all')
    athe = mcha.df.energy.sum()
    he00 = total_hit_energy(mcHits,
                                event_slice = slice(0,0),
                                particle_slice = slice(None, None))
    ev = he00.total_hit_energy.values[0]
    assert np.allclose(ev, athe)
