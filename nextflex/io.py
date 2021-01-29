import numpy as np
import pandas as pd
import networkx as nx
import json

from nextflex.reco_functions import GTrack
from nextflex.reco_functions import GTracks

from typing      import List, Tuple, Dict
from dataclasses import asdict


def write_event_gtracks_json(egtrk : List[GTracks], path : str):
    """
    Writes a list of gtracks to a file using json format

    """
    devt = {}
    for ievt, gtrks in enumerate(egtrk): # loop over events
        # Create a dictionary of json objects (from networkx objects)
        dgtrk = {j:nx.node_link_data(gtrks[j].gt)\
             for j, _ in enumerate(gtrks)}
        #print(f'ievt = {ievt}', trk0 = {dgtrk[0]}')

        # create dictionaries for all other fields
        devid = {j : int(gtrks[j].event_id)   for j, _ in enumerate(gtrks)}
        dvbin = {j : float(gtrks[j].voxel_bin)  for j, _ in enumerate(gtrks)}
        dcont = {j : float(gtrks[j].contiguity) for j, _ in enumerate(gtrks)}

        # print(f'ievt = {ievt}', devid0 = {devid[0]}')
        # print(f'ievt = {ievt}', dvbin = {dvbin[0]}')
        # print(f'ievt = {ievt}', dcont = {dcont[0]}')
        #

        # create a dict of json objects corresponding to the GTrack
        dGtrk = {"gtrk" : dgtrk, "event_id" : devid,
                 "voxel_bin" : dvbin,
                 "contiguity" : dcont}

        #print(f'ievt = {ievt}', dGtrk = {dGtrk}')

        # and add to the dictionary of events
        devt[ievt] = dGtrk

    # write to disk
    with open(path, 'w') as fp:
        json.dump(devt, fp)


def load_event_gtracks_json(path : str)->List[GTracks]:
    """
    Loads a list of gtracks in json format from file

    """
    # First load the json object from file

    with open(path) as json_file:
        jdevt = json.load(json_file)

    # then recreate the list of GTracks
    ETRKS = []
    for _, dGtrk in jdevt.items():
        GTRKS = []
        dgtrk      = dGtrk['gtrk']
        event_id   = dGtrk['event_id']
        voxel_bin  = dGtrk['voxel_bin']
        contiguity = dGtrk['contiguity']

        for it, value in dgtrk.items():
            gt = nx.node_link_graph(value)
            GTRKS.append(GTrack(gt,
                                event_id[it],
                                voxel_bin[it],
                                contiguity[it]))
        ETRKS.append(GTRKS)
    return ETRKS


def write_event_gtracks_pd(gtrksEvt : List[GTracks], path : str):
    """
    Writes a list of gtracks to a file using pandas format

    """


    index_tuples = []
    data = []
    for evt_number, gtrks in enumerate(gtrksEvt):
        for trk_number, gt in enumerate(gtrks):
            vb1 = voxels_in_blob(gt, rb, extreme ='e1')
            vb2 = voxels_in_blob(gt, rb, extreme ='e2')
            index_tuples.append((evt_number, trk_number))
            data.append({'event_id'    : gt.event_id,
                         'track_id'    : trk_number,
                         'n_voxels'    : len(gt.voxels),
                         'trak_length' : gt.length,
                         'energy'      : gt.voxels_df.energy.sum() / keV,
                         'x_e1'        : gt.extrema['e1'][0],
                         'y_e1'        : gt.extrema['e1'][1],
                         'z_e1'        : gt.extrema['e1'][2],
                         'energy_e1'   : gt.extrema['e1'][3],
                         'nvox_b1'     : vb1.energy.count(),
                         'energy_b1'   : blob_energy(gt, rb, extreme  ='e1'),
                         'x_e2'        : gt.extrema['e2'][0],
                         'y_e2'        : gt.extrema['e2'][1],
                         'z_e2'        : gt.extrema['e2'][2],
                         'energy_e2'   : gt.extrema['e2'][3],
                         'nvox_b2'     : vb2.energy.count(),
                         'energy_b2'   : blob_energy(gt, rb, extreme  ='e2'),
                        })
    index = pd.MultiIndex.from_tuples(index_tuples,
                                      names=["evt_number","trk_number"])

    return pd.DataFrame(data,index)


def save_to_JSON(acls, path, numpy_convert=True):
    """
    Converts reco analysis data classes to JSON and writes it to path
    numpy_convert is True when the data needs to be converted
    from type numpy (this happens for lists extracted from a DF)

    """

    dcls = asdict(acls)
    if numpy_convert:
        facls = {k:[float(x) for x in v] for (k,v) in dcls.items()}
    else:
        facls = dcls

    # then write
    with open(path, 'w') as fp:
        json.dump(facls, fp)


def load_from_JSON(path):
    """
    Reads a JSON object as a dict

    """
    with open(path) as json_file:
        jdict = json.load(json_file)
    return jdict
