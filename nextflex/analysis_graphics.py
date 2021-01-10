import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tics.graphics_reco import plot_gtrack
from  tics.system_of_units import *


def plot_single_tracks_list(gtEvent,
                       trackList,
                       rb,
                       autorange = True,
                       weight    = 'energy',
                       xrange    = (-500,500),
                       yrange    = (-500,500),
                       zrange    = (0,1000),
                       nplots    =10,
                       figsize   = (14,10),
                       fontsize  =10):
    """
    Plots up to nplots gtracks of list trackList.
    Use to fully display GTracks including end-voxels
    and blobs. It will display only the first track
    of the GTEvent.

    """

    np = 0
    for event_number in trackList:
        print(f"event number in gtEvent list  = {event_number}")
        if np > nplots:
            break
        np+=1
        gtrks = gtEvent[event_number]
        plot_gtrack(gtrks[0], rb,
                   autorange, weight, xrange, yrange, zrange,
                   figsize, fontsize)


def plot_multiple_tracks_list(gtEvent,
                              trackList,
                              autorange = True,
                              xrange    = (-500,500),
                              yrange    = (-500,500),
                              zrange    = (0,1000),
                              nplots    = 10,
                              figsize   = (14,10),
                              fontsize  =10):
    """
    Plots up to nplots gtracks of list trackList.
    Use to display multiple GTracks in an event.
    Each GTrack comes in a diferent color

    """

    np = 0
    for event_number in trackList:
        print(f"event number in gtEvent list  = {event_number}")
        gtrks = gtEvent[event_number]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        if not autorange:
            ax.set_xlim3d(*xrange)
            ax.set_ylim3d(*yrange)
            ax.set_zlim3d(*zrange)

        for i, gt in enumerate(gtrks):
            p = ax.scatter(gt.voxels_df.x, gt.voxels_df.y, gt.voxels_df.z)
        plt.show()
        np+=1
        if np > nplots:
            break