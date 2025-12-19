import argparse
import h5py as h5
import numpy as np
from popde import density_estimate as d, adaptive_kde as ad
import priors_vectorize as spin_prior
from matplotlib import use
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import PowerNorm
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u
import utils_plot as u_plot
#from cbc_pdet import gwtc_found_inj  as pdet_fit
from cbc_pdet import o123_class_found_inj_general  as pdet_fit


# Set Matplotlib parameters
rcParams.update({
    "text.usetex": True,
    "font.serif": "Computer Modern",
    "font.family": "Serif",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 18,
    "axes.labelsize": 18,
    "axes.grid": True,
    "grid.color": 'grey',
    "grid.linewidth": 1.0,
    "grid.alpha": 0.6
})


parser = argparse.ArgumentParser(description=__doc__)
# Input files
parser.add_argument('--samples-dim1', help='h5 file containing N samples for m1 for each event')
parser.add_argument('--samples-dim2', help='h5 file containing N samples of m2 for each event')
parser.add_argument('--samples-dim3', help='h5 file containing N samples of chieff for each event')
parser.add_argument('--samples-redshift', help='h5 file containing N samples of redshift for each event')
parser.add_argument('--samples-dl', help='h5 file containing N samples of dL for each event')
parser.add_argument('--samples-vt', help='h5 file containing VT calculated at each sample')
parser.add_argument('--pdet-runs', help='Observing runs to derive VT from p_det fits for, eg "o123", "o4"')
parser.add_argument('--injectionfile',  help='h5 injection file from GWTC3 public data', default='endo3_bbhpop-LIGO-T2100113-v12.hdf5')

# PE prior
parser.add_argument('--redshift-prior-power', type=float, default=2.,
                    help='PE prior to factor out is this power of (1+z)')
parser.add_argument('--pe-chieff-prior', action='store_true',
                    help='Reweight during analysis to remove PE prior over chieff. Default False')

# Rescaling factor bounds [bandwidth]
parser.add_argument('--min-bw3', default=0.01, type=float, help='Set a minimum bandwidth for the 3rd dimension')

# Buffer iterations
parser.add_argument('--buffer-start', default=200, type=int, help='Start iteration for buffer in reweighting')
parser.add_argument('--buffer-interval', default=100, type=int, help='Size of buffer')
parser.add_argument('--n-iterations', default=500, type=int, help='Total reweighting iterations after start of buffer')

# Output
parser.add_argument('--pathplot', default='./', help='public_html path for plots', type=str)
parser.add_argument('--output-filename', required=True, help='name of analysis output hdf file', type=str)

opts = parser.parse_args()  # Use as global variable

#####################################################################

# Set the prior factors correctly here before reweighting
prior_kwargs = {'redshift_prior_power': opts.redshift_prior_power}
print(f"prior powers: {prior_kwargs}")

# Cosmology
H0 = 67.9  # km/s/Mpc
omega_m = 0.3065
cosmo = FlatLambdaCDM(H0=H0, Om0=omega_m)


def preprocess_data(m1_injection, dL_injection, pe_m1, pe_dL, num_bins=10):
    """
    Preprocess data by filtering invalid entries based on distance limits.

    Args:
        m1_injection (np.ndarray): Injected primary masses.
        dL_injection (np.ndarray): Injected luminosity distances.
        pe_m1 (np.ndarray): Primary mass posterior estimates.
        pe_dL (np.ndarray): Luminosity distance posterior estimates.
        num_bins (int): Number of bins for preprocessing.

    Returns:
        tuple: Filtered primary masses, distances, and their indices.
    """
    log_m1 = np.log10(m1_injection)
    pe_log_m1 = np.log10(pe_m1)
    bins = np.linspace(log_m1.min(), log_m1.max(), num_bins + 1)

    max_dL_per_bin = np.array([
        dL_injection[(log_m1 >= bins[i]) & (log_m1 < bins[i + 1])].max()
        if ((log_m1 >= bins[i]) & (log_m1 < bins[i + 1])).any() else -np.inf
        for i in range(len(bins) - 1)
    ])

    filtered_pe_m1 = []
    filtered_pe_dL = []
    filtered_indices = []

    for i in range(len(bins) - 1):
        bin_mask = (pe_log_m1 >= bins[i]) & (pe_log_m1 < bins[i + 1])
        max_dL = max_dL_per_bin[i]
        keep_mask = bin_mask & (pe_dL <= max_dL)
        filtered_pe_m1.extend(pe_m1[keep_mask])
        filtered_pe_dL.extend(pe_dL[keep_mask])
        filtered_indices.extend(np.where(keep_mask)[0])

    print('keeping', len(filtered_indices), 'samples')
    return (
        np.array(filtered_pe_m1),
        np.array(filtered_pe_dL),
        np.array(filtered_indices)
    )


def prior_factor_function(samples, redshift_vals, redshift_prior_power):
    """
    Compute a prior factor for reweighting for chi_eff and source frame component masses.
    Apply (1+z)^power for redshift scaling.

    Args:
        samples (np.ndarray): Array of samples with shape (N, 3).
        redshift_vals (np.ndarray): Redshift values corresponding to the samples.
        redshift_prior_power (float): Power to apply as redshift prior factor.

    Returns:
        np.ndarray: Prior factor for each sample, computed as 1 / (chieff_prior * (1+z)^power).
    """
    if samples.shape[1] != 3:
        raise ValueError("Samples array must have exactly three columns.")

    if len(redshift_vals) != len(samples):
        raise ValueError("Number of redshifts must match the number of samples.")

    if opts.pe_chieff_prior:
        m1, m2, cf = samples[:, 0], samples[:, 1],  samples[:, 2]
        aMax = 0.999
        chieff_prior = spin_prior.chi_effective_prior_from_isotropic_spins(m2 / m1, aMax, cf)
    else:
        chieff_prior = 1.

    redshift_prior = (1. + redshift_vals) ** redshift_prior_power

    prior_factors = 1. / (chieff_prior * redshift_prior)
    return prior_factors


def get_reweighted_sample(rng, sample, redshiftvals, vt_vals, fpop_kde, prior_factor=prior_factor_function, prior_factor_kwargs={}):
    """
    Generate reweighted random sample/samples from the original PE samples

    Samples are reweighted based on the current estimate of the population
    distribution, detection probability or VT values and a PE prior factor, and then performs weighted random
    sampling.

    Parameters
    -----------
    sample :  array-like
        The list or array of PE samples representing a set of events or observations.

    redshiftvals : array-like
        Sample redshift values used to compute prior factors.

    vt_vals : array-like
        The sensitive volume time values for each sample, used to scale the KDE estimate.

    fpop_kde : KDE object
        A kernel density estimate (KDE) object that models the detected event distribution.

    prior_factor : callable
        A function that calculates the prior factor for each sample, typically dependent on the redshift.

    prior_factor_kwargs : dict
        Additional kwargs for the `prior_factor` function.

    Returns
    --------
    (float), (float), (array)
        Randomly selected weighted sample, its VT value and array of weights
    """
    # Ensure prior_factor_kwargs is a dictionary
    if prior_factor_kwargs is None:
        prior_factor_kwargs = {}

    # Evaluate the KDE estimate of the astrophysical population at samples
    fkde_samples = fpop_kde.evaluate_with_transf(sample) / vt_vals

    # Adjust probabilities based on the prior factor
    frate_atsample = fkde_samples * prior_factor(sample, redshiftvals, **prior_factor_kwargs)
    # Normalize
    fpop_at_samples = frate_atsample / frate_atsample.sum()

    # Select a sample using weighted random sampling
    selected_idx = rng.choice(len(sample), p=fpop_at_samples)

    return sample[selected_idx], vt_vals[selected_idx], fpop_at_samples


def buffer_reweighted_sample(rng, sample, redshiftvals, vt_vals, meanKDEevent, prior_factor=prior_factor_function, prior_factor_kwargs=None):
    """
    Generate a reweighted sample based on the average of estimated KDE vals for a set of PE samples,
    adjusting for detection probability and a possible PE prior factor.

    Parameters
    -----------
    sample : array-like
        PE samples over the KDE space for a given event.

    redshiftvals : array-like
        Redshift values corresponding to the PE samples.

    vt_vals : array-like
        Sensitive Volume * Time at each PE sample.

    meanKDEevent : array-like
        The mean KDE values for the PE samples calculated over some number of previous iterations.

    prior_factor : callable
        A function that calculates the prior factor for each sample, typically depending on redshift.

    prior_factor_kwargs : dict
        Additional kwargs for the `prior_factor` function.

    Returns
    --------
    (float), (float), (array)
        Randomly selected weighted sample, its VT value and array of weights
    """
    # Ensure prior_factor_kwargs is a dictionary
    if prior_factor_kwargs is None:
        prior_factor_kwargs = {}

    # Compute KDE probabilities divided by sensitivity
    kde_by_vt = meanKDEevent / vt_vals

    # Adjust probabilities based on the prior factor
    kde_by_vt *= prior_factor(sample, redshiftvals, **prior_factor_kwargs)

    # Normalize
    norm_meankdevals = kde_by_vt / sum(kde_by_vt)

    # Select a sample using weighted random sampling
    selected_idx = rng.choice(len(sample), p=norm_meankdevals)

    return sample[selected_idx], vt_vals[selected_idx], norm_meankdevals


def get_kde_obj_eval(sample, bs_weights, rescale_arr, alpha, input_transf=('log', 'log', 'none'), mass_symmetry=False, minbw3=opts.min_bw3):
    if bs_weights is not None:
        # Remove samples with zero bootstrap weight as they may have bad behaviour in
        # adaptive KDE (due to extremely small pilot density at the sample location)
        # and have no effect on the KDE, and to reduce compute cost
        sample = sample[bs_weights > 0., :]
        bs_weights = bs_weights[bs_weights > 0.]

    # Apply m1-m2 symmetry in the samples when making KDEs
    symm_dims = [0, 1] if mass_symmetry else None

    kde_object = ad.KDERescaleOptimization(sample, bs_weights, input_transf=input_transf, stdize=True,
        rescale=rescale_arr, symmetrize_dims=symm_dims, alpha=alpha, dim_names=['lnm1', 'lnm2', 'chieff']
    )
    dictopt, score = kde_object.optimize_rescale_parameters(
        rescale_arr, alpha, bounds=((0.01, 100), (0.01, 100), (0.01, 1./ minbw3), (0, 1)), disp=False
    )  # don't display messages
    optbwds = 1. / dictopt[0:-1]
    optalpha = dictopt[-1]

    return kde_object, optbwds, optalpha


#######################################################################
# STEP I: call the PE sample data and get Pdet on PE samples
# get VT(m1, m2, chieff)

run_opt = 'o4' if opts.pdet_runs == 'o4' else 'o3'
print(run_opt)
dmid_fun = 'Dmid_mchirp_fdmid_fspin'
emax_fun = 'emax_exp'
pdet = pdet_fit.Found_injections(dmid_fun=dmid_fun, emax_fun=emax_fun, alpha_vary=None, ini_files=None)
pdet.get_opt_params(run_opt, rescale_o3=False)
pdet.set_shape_params()
pdet.load_inj_set(run_opt)

fz = h5.File(opts.samples_redshift, 'r')
dz = fz['randdata']
fdL = h5.File(opts.samples_dl, 'r')
ddL = fdL['randdata']

f1 = h5.File(opts.samples_dim1, 'r') # m1
d1 = f1['randdata']
f2 = h5.File(opts.samples_dim2, 'r') # m2
d2 = f2['randdata']
f3 = h5.File(opts.samples_dim3, 'r') # chieff
d3 = f3['randdata']

mean1 = f1['initialdata/original_mean'][...]
mean2 = f2['initialdata/original_mean'][...]
mean3 = f3['initialdata/original_mean'][...]

eventlist = []
sampleslists1 = []
sampleslists2 = []
sampleslists3 = []
redshiftlists = []

if opts.pdet_runs == 'o1o2o3':
    pdet.runs = ['o1', 'o2', 'o3']  # Otherwise uses o1-o4 ...
    sensitivity = lambda m1, m2, chieff: \
                      pdet.total_sensitive_volume(m1, m2, chieff)
    print('using total sensitive volume for', pdet.runs)
elif opts.pdet_runs == 'o4':
    sensitivity = lambda m1, m2, chieff: \
                      pdet.sensitive_volume('o4', m1, m2, chieff, zmax=2.5)
    print('o4 approximation, pdet runs are', pdet.runs)
vth5file = h5.File(opts.samples_vt, "a")  # create if file does not exist
vtlists = []

for k in d1.keys():
    eventlist.append(k)

    # These events' PE had some 'too-distant' samples with extremely small pdet
    if (k == 'GW190719_215514_mixed-nocosmo' or k == 'GW190805_211137_mixed-nocosmo'):
        z_val = dz[k][...]
        m1_val = d1[k][...]
        # Detector frame mass for pdet
        m1det_val = d1[k][...] * (1. + z_val)
        m2_val = d2[k][...]
        m2det_val = d2[k][...] * (1. + z_val)
        dL_val = ddL[k][...]
        chieff_val = d3[k][...]

        # clean data using injection mass/dL as reference
        with h5.File(opts.injectionfile, 'r') as f:
            injection_m1 = f['injections/mass1_source'][:]
            injection_dL = f['injections/distance'][:]
        print('cleaning samples for', k)
        m1_val, dL_val, idx = preprocess_data(injection_m1, injection_dL, m1_val, dL_val)

        m2_val = m2_val[idx]
        m1det_val = m1det_val[idx]
        m2det_val = m2det_val[idx]
        chieff_val = chieff_val[idx]
        z_val = z_val[idx]
    else:
        z_val = dz[k][...]
        m1_val = d1[k][...]
        m1det_val = d1[k][...] * (1. + z_val)
        m2_val = d2[k][...]
        m2det_val = d2[k][...] * (1. + z_val)
        dL_val = ddL[k][...]
        chieff_val = d3[k][...]

    try:
        vt_val = vth5file[k][:]
        print('Got VT from file for', k)
    except:
        print('Calculating pdet for', k)
        vt_val = np.zeros(len(m1det_val))
        for i in range(len(m1det_val)):
            vt_val[i] = sensitivity(m1_val[i], m2_val[i], chieff=chieff_val[i])
        vth5file.create_dataset(k, data=np.array(vt_val))

    vtlists.append(vt_val)
    sampleslists1.append(m1_val)
    sampleslists2.append(m2_val)
    sampleslists3.append(chieff_val)
    redshiftlists.append(z_val)

f1.close()
f2.close()
f3.close()
fz.close()
fdL.close()
vth5file.close()

flat_samples1 = np.concatenate(sampleslists1).flatten()
flat_samples2 = np.concatenate(sampleslists2).flatten()
flat_samples3 = np.concatenate(sampleslists3).flatten()
flat_vtlist = np.concatenate(vtlists).flatten()
print("min max m1 =", np.min(flat_samples1), np.max(flat_samples1))
print("min max m2 =", np.min(flat_samples2), np.max(flat_samples2))
print("min max chieff =", np.min(flat_samples3), np.max(flat_samples3))

# 1d histograms
for i, tupl in enumerate(zip([flat_samples1, flat_samples2, flat_samples3], [True, True, False])):
    samp = tupl[0]
    if tupl[1]: samp = np.log10(samp)
    plt.hist(samp, bins=100, histtype='step')
    if tupl[1]: plt.xlabel(r'log$_{10} ' + fr'x_{i+1}$')
    else: plt.xlabel(fr'$x_{i+1}$')
    plt.savefig(opts.pathplot+f'sample_hist_x{i+1}.png')
    plt.close()

# 2d histograms : color ~ nsamples^-0.5
plt.figure(figsize=(8, 6))
plt.hist2d(np.log10(flat_samples1), np.log10(flat_samples2), bins=100, norm=PowerNorm(gamma=0.5))
plt.xlabel(r'log$_{10} x_1$'); plt.ylabel(r'log$_{10} x_2$')
plt.savefig(opts.pathplot+f'sample_hist_x1x2.png'); plt.close()

plt.figure(figsize=(8, 6))
plt.hist2d(np.log10(flat_samples1), flat_samples3, bins=100, norm=PowerNorm(gamma=0.5))
plt.xlabel(r'log$_{10} x_1$'); plt.ylabel(r'$x_3$')
plt.savefig(opts.pathplot+f'sample_hist_x1x3.png'); plt.close()

plt.figure(figsize=(8, 6))
plt.hist2d(np.log10(flat_samples2), flat_samples3, bins=100, norm=PowerNorm(gamma=0.5))
plt.xlabel(r'log$_{10} x_2$'); plt.ylabel(r'$x_3$')
plt.savefig(opts.pathplot+f'sample_hist_x2x3.png'); plt.close()

#exit()
# Scatter plots for pdet
def eta_from_mass1_mass2(mass1, mass2):
    return mass1 * mass2 / (mass1 + mass2)**2.

def mchirp_from_mass1_mass2(mass1, mass2):
    return eta_from_mass1_mass2(mass1, mass2)**(3./5) * (mass1 + mass2)

Mchirp = mchirp_from_mass1_mass2(flat_samples1, flat_samples2)
u_plot.plot_pdet_scatter(Mchirp, flat_samples3, flat_vtlist, xlabel=r'$\mathcal{M}$', ylabel=r'$\chi_\mathrm{eff}$', title=r'$VT$', save_name="VT_mc_chieff_scatter.png", pathplot=opts.pathplot)
u_plot.plot_pdet_scatter(flat_samples2/flat_samples1, flat_samples3, flat_vtlist, xlabel=r'$q$', ylabel=r'$\chi_\mathrm{eff}$', title=r'$VT$', save_name="VT_q_chieff_scatter.png", pathplot=opts.pathplot)

# 3D scatter plot
u_plot.plot_pdet_3Dscatter(flat_samples1, flat_samples2, flat_samples3, flat_vtlist, save_name="pdet_m1m2chieff_3Dscatter.png", pathplot=opts.pathplot)


##########################################
sampleslists = np.vstack((flat_samples1, flat_samples2, flat_samples3)).T
mean_sample = np.vstack((mean1, mean2, mean3)).T

### Iterative reweighting algorithm
discard = opts.buffer_start   # how many iterations to discard
Nbuffer = opts.buffer_interval # how many previous iterations to average over in reweighting

init_rescale = [3., 3., 3.]
init_alpha = 0.5

# First mean samples KDE (no weights)
current_kde, bws, alp = get_kde_obj_eval(mean_sample, None, init_rescale, init_alpha, mass_symmetry=True, input_transf=('log', 'log', 'none'), minbw3=opts.min_bw3)
print('Initial opt parameters', bws, alp)

def Neff(weights):
    """
    Effective sample size in importance sampling with given weights
    """
    w = np.array(weights)
    return w.sum() ** 2. / (w ** 2.).sum()

### Iterative reweighting algorithm

# Save KDE parameters for each subsequent iteration in HDF file
frateh5 = h5.File(opts.output_filename + '_kde_iteration.hdf5', 'a')

# Start by recording options
for oname, oval in vars(opts).items():
    frateh5.attrs.create(oname, oval)

# Store iteration statistics
iterbwx = []
iterbwy = []
iterbwz = []
iteralp = []
iterminneff = []

discard = opts.buffer_start   # how many iterations to discard
Nbuffer = opts.buffer_interval # how many previous iterations to average over in reweighting

# Initialize buffer to store last Nbuffer iterations of f(samples) for each event
num_events = len(mean1)
buffers = [[] for _ in range(num_events)]

rng = np.random.default_rng()
for i in range(opts.n_iterations + discard):  # eg 500 + 200
    # Take 1 reweighted PE sample per event and weight it in KDE evaluation and optimization by a Poisson bootstrap factor
    rwsamples = []
    rw_neff = []  # Effective number of samples from fpop/prior weighting
    rwvt_vals = []
    boots_weights = []
    # Loop over events
    for eventid, (samplem1, samplem2, sample3, redshiftvals, vt_k) in \
            enumerate(zip(sampleslists1, sampleslists2, sampleslists3, redshiftlists, vtlists)):
        event_boots_weight = rng.poisson(1)
        if event_boots_weight == 0:  # Immediately discard cases with zero weight
            continue
        samples = np.vstack((samplem1, samplem2, sample3)).T
        # Determine weights for next draw by evaluating previous KDE on all samples
        event_kde = current_kde.evaluate_with_transf(samples)
        buffers[eventid].append(event_kde)

        if i < discard + Nbuffer:  # eg if less than 200 + 100
            rwsample, rwvt_val, rweights = get_reweighted_sample(rng, samples, redshiftvals, vt_k, current_kde, prior_factor_kwargs=prior_kwargs)
        else:  # start to reweight based on buffer
            # Use average of previous Nbuffer KDE evaluations on samples
            means_kde_event = np.mean(buffers[eventid][-Nbuffer:], axis=0)
            rwsample, rwvt_val, rweights = buffer_reweighted_sample(rng, samples, redshiftvals, vt_k, means_kde_event, prior_factor_kwargs=prior_kwargs)
        rwsamples.append(rwsample)
        rw_neff.append(Neff(rweights))
        rwvt_vals.append(rwvt_val)
        boots_weights.append(event_boots_weight)

    # Reassign current KDE to optimized estimate for this iteration
    current_kde, optbw, optalp = get_kde_obj_eval(np.array(rwsamples), np.array(boots_weights), init_rescale, init_alpha, mass_symmetry=True, input_transf=('log', 'log', 'none'), minbw3=opts.min_bw3)
    # Get perpoint bandwidths
    perpointbws = current_kde.bandwidth[:len(rwsamples)]
    print("opt bw", optbw, "opt alpha", optalp, 'Neff min/max', np.min(rw_neff), np.max(rw_neff), 'min Neff eventid', np.argmin(rw_neff))

    group = frateh5.create_group(f'iteration_{i}')

    # Save iteration data
    group.create_dataset('rwsamples', data=np.array(rwsamples))
    group.create_dataset('perpoint_bws', data=np.array(perpointbws))
    group.create_dataset('rwvt_vals', data=np.array(rwvt_vals))
    group.create_dataset('bootstrap_weights', data=np.array(boots_weights))
    group.create_dataset('alpha', data=optalp)
    group.create_dataset('bwx', data=optbw[0])
    group.create_dataset('bwy', data=optbw[1])
    group.create_dataset('bwz', data=optbw[2])
    frateh5.flush()

    iterbwx.append(optbw[0])
    iterbwy.append(optbw[1])
    iterbwz.append(optbw[2])
    iteralp.append(optalp)
    iterminneff.append(np.min(rw_neff))

    if i > 1 and i % 100==0:
        print(i)
        u_plot.histogram_bw(iterbwx[-Nbuffer:], 'bwx', opts.pathplot, tag=i)
        u_plot.histogram_bw(iterbwy[-Nbuffer:], 'bwy', opts.pathplot, tag=i)
        u_plot.histogram_bw(iterbwz[-Nbuffer:], 'bwz', opts.pathplot, tag=i)
        u_plot.histogram_bw(iteralp[-Nbuffer:], 'alpha', opts.pathplot, tag=i)

frateh5.close()

u_plot.bw_correlation(iterbwx, discard, 'bwx', opts.pathplot)
u_plot.bw_correlation(iterbwy, discard, 'bwy', opts.pathplot)
u_plot.bw_correlation(iterbwz, discard, 'bwz', opts.pathplot)
u_plot.bw_correlation(iteralp, discard, 'alpha', opts.pathplot, log=False)
u_plot.bw_correlation(iterminneff, discard, r'min(Neff)', opts.pathplot)

