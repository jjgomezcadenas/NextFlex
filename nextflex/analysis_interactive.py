import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tics.graphics_reco import plot_gtrack
from tics.graphics_reco import sphere
from  tics.system_of_units import *

from nextflex.reco_functions import blob_energy
from nextflex.reco_functions import voxels_in_blob


def set_fonts(ax, fontsize=20):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)


def blob_selection_efficiency(bbdf, e1df, eb_cut):
    """
    bbdf and e1df are datframe holding blob information.

    Computes efficiency and figures of merit for
    signal and background, given eb_cut.

    """
    def select(df, ebcut):
        eb1 = df[df.eb1 > ebcut]
        return eb1[eb1.eb2 > ebcut]

    blobs_eff_bb    = len(select(bbdf, eb_cut)) / len(bbdf)
    blobs_eff_1e    = len(select(e1df, eb_cut)) / len(e1df)
    f1 = blobs_eff_bb / blobs_eff_1e
    f2 = blobs_eff_bb / np.sqrt(blobs_eff_1e)

    return blobs_eff_bb, blobs_eff_1e, f1, f2


def blobs_figure_of_merit(bbdf,
                          e1df,
                          eb_range = (250,650),
                          npoints  = 10,
                          plotting = True,
                          figsize=(14,7)):

    """
    bbdf and e1df are datframe holding blob information.

    Computes curves of efficiency and figures of merit for
    signal and background.

    """
    def select(df, ebcut):
        eb1 = df[df.eb1 > ebcut]
        return eb1[eb1.eb2 > ebcut]


    eb_cut = np.linspace(*eb_range,npoints)

    eBB = np.array([len(select(bbdf, ecut)) / len(bbdf)\
                    for ecut in eb_cut])
    e1E = np.array([len(select(e1df, ecut)) / len(e1df) \
                    for ecut in eb_cut])

    fm1 = np.divide(eBB, e1E, out=np.zeros_like(eBB), where=e1E!=0)
    fm2 = np.divide(eBB, np.sqrt(e1E), out=np.zeros_like(eBB), where=e1E!=0)
    #fm1 = eBB / e1E
    #fm2 = eBB / np.sqrt(e1E)

    if plotting:
        fig = plt.figure(figsize=figsize)
        ax      = fig.add_subplot(2, 2, 1)
        plt.plot(eb_cut, eBB)
        plt.xlabel("Eb cut (keV)")
        plt.ylabel("bb0nu efficiency (%)")
        ax      = fig.add_subplot(2, 2, 2)
        plt.plot(eb_cut, e1E)
        plt.xlabel("Eb cut (keV)")
        plt.ylabel("1e efficiency (%)")
        ax      = fig.add_subplot(2, 2, 3)
        plt.plot(eb_cut, fm1)
        plt.xlabel("Eb cut (keV)")
        plt.ylabel("S/N ")
        ax      = fig.add_subplot(2, 2, 4)
        plt.plot(eb_cut, fm2)
        plt.xlabel("Eb cut (keV)")
        plt.ylabel("S/Sqrt(N) ")
        plt.tight_layout()
        plt.show()

    return eb_cut, eBB, e1E, fm1, fm2


def emin_emax(bbdf, e1df, minval=10):
    """
    bbdf and e1df are datframe holding blob information.
    Computes the range of energy blob to scan.

    """
    def select_e(values, bins, lb):
        for i, mbin in enumerate(values):
            if mbin > minval:
                lb = i
                break
        return lb, bins[lb]

    def select(df):
        values, bins = np.histogram(df.eb1)
        #print(values, bins)
        lb = 0
        lb, emin = select_e(values, bins, lb)
        #print(f"emin = {emin}")

        lb = -1
        reversed_values = values[::-1]
        reversed_bins   = bins[::-1]
        lb, emax = select_e(reversed_values, reversed_bins, lb)
        #print(f"emax = {emax}")

        return emin, emax

    bbemin, bbemax = select(bbdf)
    e1emin, e1emax = select(e1df)

    emin = np.max([bbemin,e1emin])
    emax = np.min([bbemax,e1emax])
    return emin, emax


def draw_blobs_energy_and_distances(bbdf,
                                    e1df,
                                    eb_cut,
                                    d1_cut,
                                    eb_range=(0,500),
                                    figsize=(10,10)):
    """
    bbdf and e1df are datframe holding blob information.
    d1,2 are the distances betweem true and reconstructed voxels
    eb1,2 are the energy of the blobs


    """

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(2, 2, 1)
    plt.scatter(bbdf.d1, bbdf.d2)
    ax.set_xlim([0,50])
    ax.set_ylim([0,50])
    ax.set_xlabel('d1 (mm)')
    ax.set_ylabel('d2 (mm)')
    plt.axvline(x = d1_cut)
    plt.axhline(y = d1_cut)


    ax  = fig.add_subplot(2, 2, 2)
    plt.scatter(bbdf.eb1, bbdf.eb2)
    ax.set_xlim(*eb_range)
    ax.set_ylim(*eb_range)
    ax.set_xlabel('eb1 (keV)')
    ax.set_ylabel('eb2 (keV)')
    plt.axvline(x = eb_cut)
    plt.axhline(y = eb_cut)

    ax  = fig.add_subplot(2, 2, 3)
    plt.scatter(e1df.d1, e1df.d2)
    ax.set_xlim([0,10])
    ax.set_ylim([0,10])
    ax.set_xlabel('d1 (mm)')
    ax.set_ylabel('d2 (mm)')
    plt.axvline(x = d1_cut)
    plt.axhline(y = d1_cut)


    ax  = fig.add_subplot(2, 2, 4)
    plt.scatter(e1df.eb1, e1df.eb2)
    ax.set_xlim(*eb_range)
    ax.set_ylim(*eb_range)
    ax.set_xlabel('eb1 (keV)')
    ax.set_ylabel('eb2 (keV)')

    plt.axvline(x = eb_cut)
    plt.axhline(y = eb_cut)
    plt.tight_layout()
    plt.show()


def draw_blobs_distances(bbdf, e1df, d1='d1', d2 = 'd2',
                         range_d1 = (0,50),
                         range_d2 = (0,50),
                         figsize=(10,10)):
    """
    bbdf and e1df are datframe holding blob information.
    d1,2 are the distances betweem true and reconstructed voxels

    """

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 2, 1)
    plt.scatter(bbdf[d1], bbdf[d2])
    ax.set_xlim(*range_d1)
    ax.set_ylim(*range_d2)
    ax.set_xlabel(f'{d1} (mm)')
    ax.set_ylabel(f'{d2} (mm)')

    ax  = fig.add_subplot(1, 2, 2)
    plt.scatter(e1df[d1], e1df[d2])
    ax.set_xlim(*range_d1)
    ax.set_ylim(*range_d2)
    ax.set_xlabel(f'{d1} (mm)')
    ax.set_ylabel(f'{d2} (mm)')

    plt.tight_layout()
    plt.show()


def draw_blobs_energy(bbdf,
                      e1df,
                      eb_cut,
                      e1 = 'eb1',
                      e2 = 'eb2',
                      eb_range=(0,500),
                      figsize=(10,10)):
    """
    bbdf and e1df are datframe holding blob information.
    eb1,2 are the energy of the blobs


    """

    fig = plt.figure(figsize=figsize)

    ax  = fig.add_subplot(1, 2, 1)
    plt.scatter(bbdf[e1], bbdf[e2])
    ax.set_xlim(*eb_range)
    ax.set_ylim(*eb_range)
    ax.set_xlabel('eb1 (keV)')
    ax.set_ylabel('eb2 (keV)')
    plt.axvline(x = eb_cut)
    plt.axhline(y = eb_cut)

    ax  = fig.add_subplot(1, 2, 2)
    plt.scatter(bbdf[e1], bbdf[e2])
    ax.set_xlim(*eb_range)
    ax.set_ylim(*eb_range)
    ax.set_xlabel('eb1 (keV)')
    ax.set_ylabel('eb2 (keV)')

    plt.axvline(x = eb_cut)
    plt.axhline(y = eb_cut)
    plt.tight_layout()
    plt.show()


def rblob_optimisation(bbdf, e1df, rblobs, verbose = False):
    """
    bbdf and e1df are datframe holding blob information.
    rblobs is an array or blob radius.
    For each blob radius find the optimal figure of merit and return
    the values of energy cut, signal and background efficiency and
    background rejection corresponding to the optimal figure of merit

    """
    EBCUT = []
    FMAX  = []
    BBEFF = []
    E1EFF = []
    SN = []

    for r in rblobs:
        bbr = bbdf[bbdf.rb==r]
        e1r = e1df[e1df.rb==r]
        emin, emax = emin_emax(bbr, e1r, minval=10)
        ebcut, eBB, e1E, fm1, fm2 = blobs_figure_of_merit(bbr, e1r, eb_range = (emin,emax), npoints=20,
                                               plotting = False, figsize=(14,7))

        fmax = np.max(fm2)
        ebct  = ebcut[np.argmax(fm2)]
        bbeff = eBB[np.argmax(fm2)]
        e1eff = e1E[np.argmax(fm2)]
        ston  = fm1[np.argmax(fm2)]

        EBCUT.append(ebct)
        FMAX.append(fmax)
        BBEFF.append(bbeff)
        E1EFF.append(e1eff)
        SN.append(ston)

        if verbose:
            print(f"""
            for rblob = {r}
            best figure of merit is {fmax:5.2f}
            cut at {ebct:5.2f} keV
            efficiency for bb = {bbeff:5.3f}
            efficiency for 1e = {e1eff:5.3f}

            """)

    return FMAX, SN, EBCUT, BBEFF, E1EFF


def plot_figures_of_merit(bbdf, e1df, rblobs):
    """
    bbdf and e1df are datframe holding blob information.
    Computes and plots figures of merits for different
    blob radius (in the list rblobs)

    """
    fMax, StoN, eBcut, bbEff, e1Eff = rblob_optimisation(bbdf,
                                                   e1df, rblobs, verbose=False)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(2,2,1)
    p = ax.scatter(rblobs, eBcut, cmap='coolwarm', c=bbEff)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('signal efficiency')
    ax.set_xlabel('blob radius (mm)')
    ax.set_ylabel('blob cut (keV)')

    ax = fig.add_subplot(2,2,2)
    p = ax.scatter(rblobs, eBcut, cmap='coolwarm', c=e1Eff)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('background efficiency')
    ax.set_xlabel('blob radius (mm)')
    ax.set_ylabel('blob cut (keV)')

    ax = fig.add_subplot(2,2,3)
    p = ax.scatter(rblobs, eBcut, cmap='coolwarm', c=StoN)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('S/N')
    ax.set_xlabel('blob radius (mm)')
    ax.set_ylabel('blob cut (keV)')

    ax = fig.add_subplot(2,2,4)
    p = ax.scatter(rblobs, eBcut, cmap='coolwarm', c=fMax)
    cb = fig.colorbar(p, ax=ax)
    cb.set_label('Figure of merit')
    ax.set_xlabel('blob radius (mm)')
    ax.set_ylabel('blob cut (keV)')

    plt.tight_layout()
    plt.show()


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
        eb1, eb2, db1, db2, ev1, ev2, dv1, dv2 = i_gtrack_and_true_extrema(gtrk,
                                                       trE,
                                                       rb,
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
                       'd2'       : db2,
                       'dv1'      : dv1,
                       'dv2'      : dv2,
                       'ebv1'     : ev1,
                       'ebv2'     : ev2})


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

        #plt.show()

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
        Energy of the blobs for smaller radius:
        eb1 = {eb1[0]:5.1f} keV
        eb2 = {eb2[0]:5.1} keV
        Energy of the blobs for larger radius:
        eb1 = {eb1[-1]:5.1f} keV
        eb2 = {eb2[-1]:5.1} keV

        """)

    def print_vox_blb():
        print(f"""

        max energy voxel inside blob1 {emB1} keV
        index max energy voxel inside  blob1 {imB1} mm
        max energy voxel inside blob2 {emB2} keV
        index max energy voxel inside blob2 {imB2} mm

        distance to max B1 = {voxB1d}, energy = {ebb1} keV
        distance to max B2 = {voxB2d}, energy = {ebb2} keV

        selected distance B1  = {vd1}, energy = {ebv1} keV
        selected distance B2 = {vd2}, energy = {ebv2} keV

        for r = {rbl[-1]} mm:
            eb1 = {eb1[-1]} keV
            eb2 = {eb2[-1]} keV
        """)


    def plot_vox_blb():
        ax = fig.add_subplot(222,  projection='3d')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('z (mm)')

        p = ax.scatter(t1[0] - voxB1.x, t1[1] - voxB1.y, t1[2] - voxB1.z,
                       cmap='coolwarm', c=(voxB1.energy / keV))
        cb = fig.colorbar(p, ax=ax)
        cb.set_label('Energy (keV)')

        ax = fig.add_subplot(223,  projection='3d')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')

        p = ax.scatter(t2[0] - voxB2.x, t2[1] -  voxB2.y, t2[2] - voxB2.z,
                       cmap='coolwarm', c=(voxB2.energy / keV))
        cb = fig.colorbar(p, ax=ax)
        cb.set_label('Energy (keV)')

        #plt.show()


    hits = gt.voxels
    e1, e2, t1, t2, d1, d2 = get_extrema_and_distance_between_extrema(gt,
                                                                 trueExtrema)
    #xrange, yrange, zrange = get_ranges(t1, t2)
    scale  = rangeshift
    xrange =(t1[0] - scale * gt.length, t1[0] + scale * gt.length)
    yrange =(t1[1] - scale * gt.length, t1[1] + scale * gt.length)
    zrange =(t1[2] + scale * gt.length, t1[2] - scale * gt.length)

    eb1 = [blob_energy(gt, rb, 'e1', unit=keV) for rb in rbl]
    eb2 = [blob_energy(gt, rb, 'e2', unit=keV) for rb in rbl]

    voxB1 = voxels_in_blob(gt, rbl[-1], extreme ='e1')
    voxB2 = voxels_in_blob(gt, rbl[-1], extreme ='e2')

    eB1  = voxB1.energy.values / keV
    emB1 = np.max(eB1)
    imB1 = np.argmax(eB1)

    eB2  = voxB2.energy.values / keV
    emB2 = np.max(eB2)
    imB2 = np.argmax(eB2)

    voxB1max = voxB1.iloc[imB1]
    voxB2max = voxB2.iloc[imB2]

    voxB1d = np.linalg.norm(voxB1max.values[0:3] - e1)
    voxB2d = np.linalg.norm(voxB2max.values[0:3] - e2)

    vd1 = np.min([voxB1d, rbl[-1]])
    vd2 = np.min([voxB2d, rbl[-1]])

    ebb1 = blob_energy(gt, voxB1d, 'e1', unit=keV)
    ebb2 = blob_energy(gt, voxB2d, 'e2', unit=keV)

    ebv1 = blob_energy(gt, vd1, 'e1', unit=keV)
    ebv2 = blob_energy(gt, vd2, 'e2', unit=keV)

    if interactive:
        print_setup()
        print_vox_blb()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        set_fonts(ax, fontsize)
        plot_setup()
        #plot_vox_blb()
        plt.show()

    return eb1, eb2, d1, d2, ebv1, ebv2, voxB1d, voxB2d
