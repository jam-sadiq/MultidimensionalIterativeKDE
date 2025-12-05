import numpy as np
import h5py
import argparse
import utils_plot as u_plot

parser = argparse.ArgumentParser()
parser.add_argument('--rate-m1-chieff-files', nargs='*')
parser.add_argument('--rate-m2-chieff-files', nargs='*')
parser.add_argument('--rate-m1-m2-files', nargs='*')
parser.add_argument('--tag')
parser.add_argument('--pathplot')
args = parser.parse_args()

m1arr = None
m1mesh = None
m1cfmesh = None
m2arr = None
m2mesh = None
m2cfmesh = None
cfarr = None
cfmesh = None

m1_slice_values = np.array([10, 15, 20, 25, 35, 45, 55, 65])
m2_slice_values = m1_slice_values * 2./3.


def get_rate_m_oneD(m1_query, m2_query, Rate):
    ratem1 = np.zeros(len(m1_query))
    ratem2 = np.zeros(len(m2_query))
    for xid, m1 in enumerate(m1_query):
        y_valid = m2_query <= m1_query[xid]  # Only accept points with y <= x
        rate_vals = Rate[y_valid, xid]
        ratem1[xid] = simpson(rate_vals, m2_query[y_valid])
    for yid, m2 in enumerate(m2_query):
        x_valid = m1_query >= m2_query[yid]  
        rate_vals = Rate[x_valid, yid]
        ratem2[yid] = simpson(rate_vals, m1_query[x_valid])
    return ratem1, ratem2

#"""
# m1-m2
m1m2_iters = []
m1m2_meanchi_iters = []
m1m2_stdchi_iters = []
m1_iters = []
m2_iters = []
for f in args.rate_m1_m2_files:
    from scipy.integrate import simpson
    with h5py.File(f, 'r') as rate:
        if m1arr is None:  # read in grid values once
            m1mesh = rate['M1mesh'][...]
            m2mesh = rate['M2mesh'][...]
            m1arr = m1mesh[:, 0]
            m2arr = m2mesh[0, :]
        for k in rate.keys():
            if (('ratem1m2' in k or 'rate_m1m2' in k) and 'iter' in k):
                r = rate[k][...]
                m1m2_iters.append(r)
                try:
                    # mean rate-weighted chieff at m1m2 points
                    mchi = rate[k.replace('m1m2', 'chim1m2')][...] / r
                    # variance of chieff at m1m2 points
                    varchi = (rate[k.replace('m1m2', 'chisqm1m2')][...] / r) - mchi ** 2
                    m1m2_meanchi_iters.append(mchi)
                    m1m2_stdchi_iters.append(varchi ** 0.5)
                except KeyError as ke:
                    print(ke)
                   
                rate1, rate2 = get_rate_m_oneD(m1arr, m2arr, rate[k])
                m1_iters.append(rate1)
                m2_iters.append(rate2)
            else:
                continue

m1m2_iters = np.array(m1m2_iters)
print(m1m2_iters.shape)
m1m2_rate_med = np.percentile(m1m2_iters, 50, axis=0)
m1m2_mean_med = np.percentile(m1m2_meanchi_iters, 50, axis=0)
m1m2_std_med = np.percentile(m1m2_stdchi_iters, 50, axis=0)

u_plot.m1m2_contour(np.array([]), np.array([]), m1mesh, m2mesh, m1m2_rate_med, timesM=True, itertag=args.tag, pathplot=args.pathplot, plot_name='Rate')
u_plot.color_m1m2_plot(np.array([]), np.array([]), m1mesh, m2mesh, m1m2_mean_med, m1m2_rate_med, timesM=True, itertag=args.tag, pathplot=args.pathplot, plot_name='meanchi')
u_plot.color_m1m2_plot(np.array([]), np.array([]), m1mesh, m2mesh, m1m2_std_med, m1m2_rate_med, timesM=True, itertag=args.tag, pathplot=args.pathplot, plot_name='stdchi')

#exit()
u_plot.oned_rate_mass(m1arr, m2arr, m1_iters, m2_iters, tag=args.tag, pathplot=args.pathplot)

#exit()
"""

# m1-chieff
#"""
m1cf_iters = []
for f in args.rate_m1_chieff_files:
    with h5py.File(f, 'r') as rate:
        if m1cfmesh is None or cfarr is None:  # read in grid values once
            m1cfmesh = rate['Mmesh'][...]
            cfmesh = rate['CFmesh'][...]
            m1arr = m1cfmesh[:, 0]
            cfarr = cfmesh[0, :]
        for k in rate.keys():
            if ('rate' in k and 'iter' in k):
                m1cf_iters.append(rate[k][...])
            else:
                continue

m1cf_iters = np.array(m1cf_iters)
print(m1cf_iters.shape)
print(m1cfmesh.shape, cfmesh.shape)
m1cf_median = np.percentile(m1cf_iters, 50, axis=0)
u_plot.m_chieff_contour(np.array([]), np.array([]), m1cfmesh, cfmesh, m1cf_median, timesM=True, itertag=args.tag, pathplot=args.pathplot, plot_name='Rate', xlabel=r'm_1')
#u_plot.get_m_Xieff_plot(np.array([]), np.array([]), m1cfmesh, cfmesh, m1cf_median, timesM=False, itertag=args.tag, pathplot=args.pathplot, plot_name='Rate', xlabel=r'm_1')

u_plot.chieff_offset_plot(m1arr, cfarr, np.array([8, 14, 20, 25, 35, 45, 55, 65]), m1cf_iters, offset_increment=5, m_label='m_1', tag=args.tag, pathplot=args.pathplot)
u_plot.chieff_offset_plot(m1arr, cfarr, np.array([10, 35, 60]), m1cf_iters, log=True, offset_increment=30, m_label='m_1', tag=args.tag, pathplot=args.pathplot)

#"""
#exit()

# m2-chieff:
m2cf_iters = []
for f in args.rate_m2_chieff_files:
    with h5py.File(f, 'r') as rate:
        if m2cfmesh is None or cfarr is None:  # read in grid values once
            m2cfmesh = rate['Mmesh'][...]
            cfmesh = rate['CFmesh'][...]
            m2arr = m2cfmesh[:, 0]
            cfarr = cfmesh[0, :]
        for k in rate.keys():
            if ('rate' in k and 'iter' in k):
                m2cf_iters.append(rate[k][...])
            else:
                continue

m2cf_iters = np.array(m2cf_iters)
m2cf_median = np.percentile(m2cf_iters, 50, axis=0)
u_plot.m_chieff_contour(np.array([]), np.array([]), m2cfmesh, cfmesh, m2cf_median, timesM=True, itertag=args.tag, pathplot=args.pathplot, plot_name='Rate', xlabel=r'm_2')
#u_plot.get_m_Xieff_plot(np.array([]), np.array([]), m2cfmesh, cfmesh, m2cf_median, timesM=False, itertag=args.tag, pathplot=args.pathplot, plot_name='Rate', xlabel=r'm_2')

u_plot.chieff_offset_plot(m2arr, cfarr, 2.*np.array([8, 14, 20, 25, 35, 45, 55, 65])/3., m2cf_iters, offset_increment=5, m_label='m_2', tag=args.tag, pathplot=args.pathplot)

