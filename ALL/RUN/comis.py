#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import gc

# Scientific import(s)
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_curve


# Project imports
#from adversarial.utils import mkdir, garbage_collect, JSD, MASSBINS
from adver.utilidad import mkdir, garbage_collect, JSD, MASSBINS


def central_rms (v_, sigma):
    """
    Iteratively compute the RMS of the events within `sigma` standard
    deviations.
    """
    v = np.array(v_, dtype=np.float)

    while len(v) > 1:

        # Compute number of outliers for inclusive set
        outliers = np.sum(np.abs(v - v.mean()) / v.std() > sigma)
        if outliers == 0: break

        # Compute z-distance from N-1 mean
        dist = np.zeros(len(v))
        for idx in range(len(v)):
            vtemp = np.delete(v, idx)
            std  = np.std(vtemp)
            mean = np.mean(vtemp)
            dist[idx] = np.abs(v[idx] - mean) / std
            pass

        # Delete furthest outlier
        idx = np.argmax(dist)
        v = np.delete(v, idx)
        pass

    if len(v) / float(len(v_)) >= 0.5:
        return v

    print ("[WARNING] No stable, central RMS was found.")
    return v_


@garbage_collect
def jsd_limit (data, frac, num_bootstrap=5, sigma=None):
    """
    General method to compute the statistical limit on JSD due to finite
    statistics.
    ...
    Arguments:
        data: ...
        frac: ...
        num_bootstrap: ...
        sigma: ...
    Returns:
        ...
    """

    msk = data['signal'] == 0
    jsd = list()
    for _ in range(num_bootstrap):

        # Manual garbage collection
        gc.collect()

        df1 = data.loc[msk, ['m', 'weight_test']].sample(frac=     frac, replace=True)
        df2 = data.loc[msk, ['m', 'weight_test']].sample(frac=1. - frac, replace=True)

        p, _ = np.histogram(df1['m'].values, bins=MASSBINS, weights=df1['weight_test'].values, density=1.)
        f, _ = np.histogram(df2['m'].values, bins=MASSBINS, weights=df2['weight_test'].values, density=1.)

        jsd.append(JSD(p, f))
        pass

    if sigma is not None:
        print ("jds_limit: Selecting central {} sigmas".format(sigma))
        print ("  frac: {} | jsd: {} ± {} -->".format(frac, np.mean(jsd), np.std(jsd)))
        jsd = central_rms(jsd, sigma=sigma)
        print ("  frac: {} | jsd: {} ± {}".format(frac, np.mean(jsd), np.std(jsd)))
        pass

    return jsd


def smooth_tgrapherrors (graph, ntimes=1):
    """
    ...
    """
    # Convert graph -> lists
    N = graph.GetN()
    x, y, ex, ey = list(), list(), list(), list()
    for i in range(N):
        x_, y_, = ROOT.Double(0), ROOT.Double(0)
        ex_, ey_ = None, None
        graph.GetPoint(i, x_, y_)
        ex_ = graph.GetErrorX(i)
        ey_ = graph.GetErrorY(i)
        x_, y_ = float(x_), float(y_)

        x.append(x_)
        y.append(y_)
        ex.append(ex_)
        ey.append(ey_)
        pass

    # Convert lists -> histograms
    x0 = x[0] - 0.5 * (x[1] - x[0])
    xh = np.array([x0] + list(x[1:] + 0.5 * np.diff(x)))

    h_y  = ROOT.TH1F('h_y',  "", N, xh)
    h_ey = h_y.Clone('h_ey')
    for i in range(N):
        h_y .SetBinContent(i + 1, y [i])
        h_ey.SetBinContent(i + 1, ey[i])
        pass

    # Smooth histograms
    h_y .Smooth(ntimes)
    h_ey.Smooth(ntimes)

    # Convert hists -> arrays
    y  = root_numpy.hist2array(h_y)
    ey = root_numpy.hist2array(h_ey)

    # Convert arrays -> graph
    for i in range(N):
        graph.SetPoint     (i, x [i], y [i])
        graph.SetPointError(i, ex[i], ey[i])
        pass

    return


def showsave (f):
    """
    Method decorrator for all study method, to (optionally) show and save the
    canvas to file.
    """
    def wrapper (*args, **kwargs):
        # Run study
        c, args, path = f(*args, **kwargs)

        # Save
        if args.save:
            dir = '/'.join(path.split('/')[:-1])
            mkdir(dir)
            suffix = path.split('.')[-1]
            if len(suffix) < 4:
                base = '.'.join(path.split('.')[:-1])
                c.save(base + '.eps')
                c.save(base + '.pdf')
                c.save(base + '.C')
            else:
                c.save(path)
                pass

            pass

        # Show
        if args.show:
            c.show()
            pass
        return

    return wrapper


def standardise (name):
    """Method to standardise a given filename, and remove special characters."""
    return name.lower() \
           .replace('#', '') \
           .replace('minus', '') \
           .replace(',', '__') \
           .replace('.', 'p') \
           .replace('(', '__') \
           .replace(')', '') \
           .replace('%', '') \
           .replace('=', '_')