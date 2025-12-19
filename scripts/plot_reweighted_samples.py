import argparse
import h5py as h5
import numpy as np
from matplotlib import use
use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib import rcParams
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/', 'kde_analysis'))

import utils_plot as u_plot

# Set Matplotlib parameters
rcParams.update({
    "text.usetex": True,
    "font.serif": "Computer Modern",
    "font.family": "Serif",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 13,
    "axes.labelsize": 13,
    "axes.grid": True,
    "grid.color": 'grey',
    "grid.linewidth": 1.0,
    "grid.alpha": 0.6
})

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--iterative-result', required=True)
parser.add_argument('--discard', default=0, type=int, help='discard first DISCARD iterations (default 0)')
parser.add_argument('--start-iter', type=int, help='start at iteration START_ITER after discards')
parser.add_argument('--end-iter', type=int, help='end at iteration END_ITER after discards')
parser.add_argument('--pathplot', default='./', help='public_html path for plots')
parser.add_argument('--tag', default='', help='optional extra string for plot filenames')
opts = parser.parse_args()

hdf = h5.File(opts.iterative_result, 'r')

bw1 = []
bw2 = []
bw3 = []
alp = []

flat_samples1 = []
flat_samples2 = []
flat_samples3 = []

mean1 = []
mean2 = []
mean3 = []
std1 = []
std2 = []
std3 = []

for i in range(opts.end_iter - opts.start_iter):
    it = i + opts.discard + opts.start_iter
    iter_name = f'iteration_{it}'
    if iter_name not in hdf:
        print(f"Iteration {it} not found in file.")
        continue

    samp = hdf[iter_name]['rwsamples'][...]
    samp1, samp2, samp3 = samp[:, 0], samp[:, 1], samp[:, 2]
    flat_samples1.append(samp1)
    flat_samples2.append(samp2)
    flat_samples3.append(samp3)

    mean1.append(samp1.mean())
    mean2.append(samp2.mean())
    mean3.append(samp3.mean())
    std1.append(samp1.std())
    std2.append(samp2.std())
    std3.append(samp3.std())

    bw1.append(hdf[iter_name]['bwx'][...])
    bw2.append(hdf[iter_name]['bwy'][...])
    bw3.append(hdf[iter_name]['bwz'][...])
    alp.append(hdf[iter_name]['alpha'][...])

flat_samples1 = np.concatenate(flat_samples1)
flat_samples2 = np.concatenate(flat_samples2)
flat_samples3 = np.concatenate(flat_samples3)

#myfilt = (flat_samples3 > 0.75) & (abs((flat_samples1) - 30) < 5)
#print(flat_samples1[myfilt])
#print(flat_samples2[myfilt])
#print(flat_samples3[myfilt])

tag = '_' + opts.tag if len(opts.tag) else '' 

# iteration summary statistics
plt.plot(mean1, '+', ms=3, label=r'$\bar{x}$')
plt.plot(mean2, '+', ms=3, label=r'$\bar{y}$')
plt.plot(mean3, '+', ms=3, label=r'$\bar{z}$')
plt.xlabel('Iteration')
plt.ylabel('Sample means')
plt.semilogy()
plt.grid(True)
plt.savefig(opts.pathplot+f'sample_mean_iters{tag}.png')
plt.close()

plt.plot(std1, '+', ms=3, label=r'$\bar{x}$')
plt.plot(std2, '+', ms=3, label=r'$\bar{y}$')
plt.plot(std3, '+', ms=3, label=r'$\bar{z}$')
plt.xlabel('Iteration')
plt.ylabel('Sample stds')
plt.semilogy()
plt.grid(True)
plt.savefig(opts.pathplot+f'sample_std_iters{tag}.png')
plt.close()

plt.plot(std3, bw3, '+', ms=3)
plt.semilogy()
plt.xlabel(r'std$(z)$')
plt.ylabel('bwz')
plt.savefig(opts.pathplot+f'stdz_v_bwz{tag}.png')
plt.close()

plt.plot(alp, bw3, '+', ms=3)
plt.semilogy()
plt.xlabel(r'alpha')
plt.ylabel('bwz')
plt.savefig(opts.pathplot+f'alpha_v_bwz{tag}.png')
plt.close()


# 1d histograms
for i, tupl in enumerate(zip([flat_samples1, flat_samples2, flat_samples3], [True, True, False])):
    samp = tupl[0]
    if tupl[1]: samp = np.log10(samp)
    plt.hist(samp, bins=100, histtype='step', density=True)
    if tupl[1]: plt.xlabel(r'log$_{10} ' + fr'x_{i+1}$')
    else: plt.xlabel(fr'$x_{i+1}$')
    plt.savefig(opts.pathplot+f'rwsample_hist_x{i+1}{tag}.png')
    plt.close()

# 2d histograms
plt.figure(figsize=(8, 6))
plt.hist2d(np.log10(flat_samples1), np.log10(flat_samples2), bins=100, norm=PowerNorm(gamma=0.5))
plt.xlabel(r'log$_{10} x_1$'); plt.ylabel(r'log$_{10} x_2$')
plt.savefig(opts.pathplot+f'rwsample_hist_x1x2{tag}.png'); plt.close()

plt.figure(figsize=(8, 6))
plt.hist2d(np.log10(flat_samples1), flat_samples3, bins=100, norm=PowerNorm(gamma=0.5))
plt.xlabel(r'log$_{10} x_1$'); plt.ylabel(r'$x_3$')
plt.savefig(opts.pathplot+f'rwsample_hist_x1x3{tag}.png'); plt.close()

plt.figure(figsize=(8, 6))
plt.hist2d(np.log10(flat_samples2), flat_samples3, bins=100, norm=PowerNorm(gamma=0.5))
plt.xlabel(r'log$_{10} x_2$'); plt.ylabel(r'$x_3$')
plt.savefig(opts.pathplot+f'rwsample_hist_x2x3{tag}.png'); plt.close()

