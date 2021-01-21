import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tics.graphics_reco import plot_gtrack
from tics.graphics_reco import sphere
from  tics.system_of_units import *

from nextflex.reco_functions import blob_energy


def set_fonts(ax, fontsize=20):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)


def plot_single_tracks_list(gtEvent,
                            trackList,
                            rb,
                            autorange   = True,
                            weight      = 'energy',
                            xrange      = (-500,500),
                            yrange      = (-500,500),
                            zrange      = (0,1000),
                            nplots      = 10,
                            figsize     = (14,10),
                            fontsize    = 10):
    """
    Plots up to nplots gtracks of list trackList.
    Use to fully display GTracks including end-voxels
    and blobs. It will display only the first track
    of the GTEvent.

    """

    np = 0
    for event_number in trackList:

        if np >= nplots:
            break
        np+=1
        gtrks = gtEvent[event_number]

        print(f"event id  = {gtrks[0].event_id}")

        plot_gtrack(gtrks[0], rb,
                    autorange, weight, xrange, yrange, zrange,
                    figsize, fontsize)


def single_track_interactive_analysis(gtEvent,
                                      trueExtrema,
                                      rb          = [2.5, 5., 10., 20],
                                      events_int  = 10,
                                      figsize     = (10,10),
                                      fontsize    = 10,
                                      rangeshift  = 20):
    """
    Interactive analysis

    """

    np = 0
    interactive = True
    dB = []
    index_tuples = []
    for event_number, gtrks in enumerate(gtEvent):
        if len(gtrks) > 1:
            print(f" Warning, ignoring event with more than one track")
            continue

        if np >= events_int:
            interactive = False
            #break
        np+=1
        gtrk = gtrks[0]
        energy = gtrk.voxels_df.energy.sum() / keV

        if interactive:
            print(f"event id  = {gtrk.event_id}, energy = {energy:5.2f} keV")

        trE   = trueExtrema[trueExtrema.evt_number == gtrk.event_id]
        eb1, eb2, db1, db2 = i_gtrack_and_true_extrema(gtrk, trE, rb,
                                                     interactive,
                                                     figsize,
                                                     fontsize,
                                                     rangeshift)

        for ir, rbl in enumerate(rb):
            index_tuples.append((event_number, ir))
            dB.append({'event_id' : gtrk.event_id,
                       'energy'   : energy,
                       'rb'       : rbl,
                       'eb1'      : eb1[ir],
                       'eb2'      : eb2[ir],
                       'd1'       : db1,
                       'd2'       : db2})


        index = pd.MultiIndex.from_tuples(index_tuples,
                                  names=["evt_number","rb_number"])


    return pd.DataFrame(dB, index)


def i_gtrack_and_true_extrema(gt,trueExtrema, rbl, interactive,
                              figsize, fontsize, rangeshift):
    """
    Draw a gtrack, including true Extrema and blobs

    """
    def sort_values(x1, x2):
        if x1 < x2:
            temp = x1
            x1 = x2
            x2 = temp
        return x1, x2

    def sort_te_by_distance_to_gt(pe, pg):

        d1 = np.linalg.norm(pe[0] - pg[0])
        d2 = np.linalg.norm(pe[0] - pg[1])
        if d1 < d2:
            e1 = pe[0]
            e2 = pe[1]
        else:
            e1 = pe[1]
            e2 = pe[0]
        return e1, e2

    def distance_te_gt(pe, pg):
        d1 = np.linalg.norm(pe[0] - pg[0])
        d2 = np.linalg.norm(pe[1] - pg[1])
        return d1, d2

    def plot_extrema(e1, e2):
        ax.scatter(e1[0], e1[1], e1[2], marker="d", s=250, color='black')
        ax.scatter(e2[0], e2[1], e2[2], marker="d", s=250, color='black')

    def plot_true_extrema(trE):
        t1 = trE.iloc[0]
        t2 = trE.iloc[1]

        ax.scatter(t1.x, t1.y, t1.z, marker="d", s=250, color='red')
        ax.scatter(t2.x, t2.y, t2.z, marker="d", s=250, color='red')


    def plot_blobs(e1, e2, rb):
        x1,y1,z1 = sphere(rb, e1[0], e1[1], e1[2])
        x2,y2,z2 = sphere(rb, e2[0], e2[1], e2[2])
        ax.plot_surface(x1, y1, z1, color='g', alpha=0.1, linewidth=0,
                        antialiased=False)
        ax.plot_surface(x2, y2, z2, color='g', alpha=0.1, linewidth=0,
                        antialiased=False)

    def get_extrema_and_distance_between_extrema(gt, trueExtrema):
        e1 = gt.extrema['e1']
        e2 = gt.extrema['e2']
        t1 = trueExtrema.iloc[0]
        t2 = trueExtrema.iloc[1]
        E12 = [e1[0:3], e2[0:3]]
        T12 = [t1.values[2:5], t2.values[2:5]]
        e1, e2 = sort_te_by_distance_to_gt(E12, T12)
        E12 = [e1[0:3], e2[0:3]]
        d1, d2 = distance_te_gt(E12, T12)
        return e1, e2, t1.values[2:5], t2.values[2:5], d1, d2

    def get_ranges(t1, t2):
        xt1, xt2 = sort_values(t1[0], t2[0])
        yt1, yt2 = sort_values(t1[1], t2[1])
        zt1, zt2 = sort_values(t1[2], t2[2])

        XT1 = [xt1, yt1, zt1]
        XT2 = [xt2, yt2, zt2]
        SGN = [(1, -1), (1, -1), (1,-1)]
        XR = [np.abs(xt1 - xt2), np.abs(yt1 - yt2), np.abs(zt1 - zt2) ]
        dmax = np.max(XR)
        ir = np.argmax(XR)
        pXR = [XT1[ir] + SGN[ir][0] * rangeshift,
               XT2[ir] + SGN[ir][1] * rangeshift]


        XD = {ir : pXR}
        for i in range(3):
            if i != ir:
                pXR = [XT1[i] + SGN[i][0] * 0.5 * dmax,
                       XT2[i] + SGN[ir][1] * 0.5 * dmax]
                XD[i] = pXR

        return [XD[i] for i in range(3)]

    def plot_setup():

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_xlim3d(*xrange)
        ax.set_ylim3d(*yrange)
        ax.set_zlim3d(*zrange)

        x1,y1,z1 = sphere(rbl[-1], e1[0], e1[1], e1[2])
        x2,y2,z2 = sphere(rbl[-1], e2[0], e2[1], e2[2])

        p = ax.scatter(gt.voxels_df.x, gt.voxels_df.y, gt.voxels_df.z,
                       cmap='coolwarm', c=(gt.voxels_df.energy / keV))
        cb = fig.colorbar(p, ax=ax)
        cb.set_label('Energy (keV)')
        plot_extrema(e1, e2)
        plot_blobs(e1, e2, rbl[-1])
        plot_true_extrema(trueExtrema)

        plt.show()

    def print_setup():
        print(f"""
        Reconstructed extrema:
        e1 = {e1} in mm
        e2 = {e2} in mm
        True extrema:
        t1 = {t1} in mm
        t2 = {t2} in mm
        Distance true-reco:
        d1 = {d1:5.1f}     in mm
        d2 = {d2:5.1f}     in mm
        Energy blobs:
        eb1 = {eb1} keV
        eb2 = {eb2} keV
        """)


    hits = gt.voxels
    e1, e2, t1, t2, d1, d2 = get_extrema_and_distance_between_extrema(gt,
                                                                 trueExtrema)
    xrange, yrange, zrange = get_ranges(t1, t2)

    eb1 = [blob_energy(gt, rb, 'e1', unit=keV) for rb in rbl]
    eb2 = [blob_energy(gt, rb, 'e2', unit=keV) for rb in rbl]

    if interactive:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        set_fonts(ax, fontsize)
        print_setup()
        plot_setup()

    return eb1, eb2, d1, d2


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
