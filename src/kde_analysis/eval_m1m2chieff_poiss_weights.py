import sys
import numpy as np
import argparse
import h5py as h5
from scipy.integrate import quad, simpson
from scipy.interpolate import RegularGridInterpolator
from matplotlib import use; use('agg')
from matplotlib import rcParams
from popde import density_estimate as kde, adaptive_kde as akde
import utils_plot as u_plot

#'''
# Set Matplotlib parameters for consistent plotting
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
# Input data
parser.add_argument('--iterative-result', required=True)
parser.add_argument('--samplesx1', help='h5 file containing N samples of m1 for each event')
parser.add_argument('--samplesx2', help='h5 file containing N samples of m2 for each event')
parser.add_argument('--samplesx3', help='h5 file containing N samples of chi_eff for each event')
parser.add_argument('--vt-file', required=True, help='VT grid hdf file')
parser.add_argument('--vt-multiplier', type=float, help='Multiplier to scale VTs up/down')

# Iterative options
parser.add_argument('--discard', default=100, type=int, help='discard first DISCARD iterations')
parser.add_argument('--start-iter', type=int, help='start at iteration START_ITER after discards')
parser.add_argument('--end-iter', type=int, help='end at iteration END_ITER after discards')
parser.add_argument('--integrate-kde', type=str, default='marginalized', choices=['marginalized', 'numeric'], help='KDE integration method: "marginalized" (integrate analytically) or "numeric" (3D KDE then numerical integration)')
# Plots and saving data
parser.add_argument('--pathplot', default='./', help='public_html path for plots')
parser.add_argument('--output-tag', required=True)

opts = parser.parse_args()


#################Integration functions######################
def integral_wrt_chieff(cf_mesh, cf_grid, Rate_3d):
    """
    Compute moments of chieff distribution at each point over m1, m2.
    Rate_3d is stored with indexing 'ij' way
    """
    integm1m2 = simpson(Rate_3d, x=cf_grid, axis=2)
    integchi_m1m2 = simpson(Rate_3d * cf_mesh, x=cf_grid, axis=2)
    integchisq_m1m2 = simpson(Rate_3d * cf_mesh * cf_mesh, x=cf_grid, axis=2)

    return integm1m2, integchi_m1m2, integchisq_m1m2


def get_m_chieff_rate_at_fixed_q(m1grid, m2grid, chieffgrid, Rate_3d, q=1.0):
    """
    q must be <=1 as m2 = q * m1mesh
    """
    M, _ = np.meshgrid(m1grid, chieffgrid, indexing='ij')
    m2values = q * M
    Rate2Dfixed_q = np.zeros_like(M)
    interpolator = RegularGridInterpolator((m1grid, m2grid, chieffgrid), Rate_3d, bounds_error=False, fill_value=None)
    for ix, m1val in enumerate(m1grid):
        Rate2Dfixed_q[ix, :] = interpolator((m1val, m2_values[ix, :], chieffgrid))
    return Rate2Dfixed_q


def get_rate_m_oneD(m1_query, m2_query, Rate):
    ratem1 = np.zeros(len(m1_query))
    ratem2 = np.zeros(len(m2_query))
    for xid, m1 in enumerate(m1_query):
        y_valid = m2_query <= m1_query[xid]  # Only accept points with y <= x
        rate_vals = Rate[y_valid, xid]
        #print(rate_vals, m2_query[y_valid])
        ratem1[xid] = simpson(rate_vals, m2_query[y_valid])
    for yid, m2 in enumerate(m2_query):
        x_valid = m1_query >= m2_query[yid]
        rate_vals = Rate[x_valid, yid]
        ratem2[yid] = simpson(rate_vals,  m1_query[x_valid])
    return ratem1, ratem2


def get_rate_m_chieff2D(m1_query, m2_query, Rate):
    ratem1 = np.zeros((len(m1_query), Rate.shape[2]))
    ratem2 = np.zeros((len(m2_query), Rate.shape[2]))

    # Iterate over each slice along the third dimension
    for i in range(Rate.shape[2]):
        # Extract the 2D slice
        Rate_slice = Rate[:, :, i]

        # Compute ratem1
        for xid, m1 in enumerate(m1_query):
            y_valid = m2_query <= m1_query[xid]  # Only accept points with y <= x
            rate_vals = Rate_slice[y_valid, xid]
            ratem1[xid, i] = simpson(rate_vals, x=m2_query[y_valid])

        # Compute ratem2
        for yid, m2 in enumerate(m2_query):
            x_valid = m1_query >= m2_query[yid]  # Only accept points with y <= x
            rate_vals = Rate_slice[x_valid, yid]
            ratem2[yid, i] = simpson(rate_vals, x=m1_query[x_valid])

    return ratem1, ratem2

#####################################################################
# Marginalization
def marginalize_kde_data(data_nd, per_point_bw, keep_dims,
                         input_transf=None, rescale_factors=None, weights=None,
                         dimension_names=None):
    """
    Marginalize N-dimensional KDE data.

    Parameters
    ----------
    data_nd : np.ndarray, shape (n_samples, n_dims)
        Original N-dimensional data points
    per_point_bw : np.ndarray, shape (n_samples,)
        Scalar per-point bandwidth factors
    keep_dims : list or array-like
        Indices of dimensions to KEEP (others integrated out)
    input_transf : tuple, optional
        Transformations per dimension (e.g., ('log', 'log', 'none'))
    rescale_factors : list, optional
        Rescale factors per dimension
    weights : np.ndarray, optional
        Sample weights
    dimension_names : list, optional
        Names for dimensions (e.g., ['m1', 'm2', 'chieff'])

    Returns
    -------
    dict with marginalized parameters
    """

    keep_dims = np.atleast_1d(keep_dims).astype(int)
    n_samples, n_dims = data_nd.shape
    n_keep = len(keep_dims)

    if dimension_names is None:
        dimension_names = [f"dim{i}" for i in range(n_dims)]

    # Validate keep_dims
    if np.any(keep_dims < 0) or np.any(keep_dims >= n_dims):
        raise ValueError(f"keep_dims must be in range [0, {n_dims-1}]")

    kept_names = [dimension_names[i] for i in keep_dims]

    # 1. Marginalize data (just select columns)
    data_marginalized = data_nd[:, keep_dims]

    # 2. Marginalize transformations
    if input_transf is not None:
        input_transf_marg = tuple(input_transf[i] for i in keep_dims)
    else:
        input_transf_marg = None

    # 3. Marginalize rescale factors
    if rescale_factors is not None:
        rescale_marg = [rescale_factors[i] for i in keep_dims]
    else:
        rescale_marg = None

    return {
        'data': data_marginalized,
        'bandwidth': per_point_bw,  
        'input_transf': input_transf_marg,
        'rescale': rescale_marg,
        'weights': weights,
        'kept_dims': list(keep_dims),
        'dimension_names': kept_names,
        'original_dims': n_dims
    }


def create_marginalized_kde(
    samples,
    per_point_bw,
    keep_dims,
    input_transf,
    rescale_factors,
    weights,
    m1grid,
    m2grid,
    cfgrid,
    alpha,
    group,
    dimension_names=None
):
    """
    Create and evaluate a marginalized KDE for specified dimensions.
    Uses symmetrize_dims in KDE objects when both m1 and m2 are present.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, 3)
        Original 3D samples
    per_point_bw : np.ndarray, shape (n_samples,)
        Scalar per-point bandwidth for each sample
    keep_dims : list of int
        Dimensions to keep (e.g., [0,1] for 2D, [0] for 1D, [0,1,2] for 3D)
    input_transf : tuple
        Input transformations for all dimensions
    rescale_factors : list
        Rescale factors for all dimensions
    weights : np.ndarray or None
        Sample weights
    m1grid, m2grid, cfgrid : np.ndarray
        Grid arrays for each dimension
    alpha : float
        Adaptive bandwidth parameter (only used if 'perpoint_bws' not in group)
    group : h5py.Group
        HDF5 group containing 'perpoint_bws' key
    dimension_names : list, optional
        Names of dimensions (default: ['m1', 'm2', 'chieff'])

    Returns
    -------
    dict with keys:
        'kde_values': Evaluated KDE on grid of remaining dimentions
        'grid_shape': Shape of the grid
        'eval_samples': Evaluation samples for KDE
        'marginalized_result': Output from marginalize_kde_data
        'kde_object': The trained KDE object
    """

    if dimension_names is None:
        dimension_names = ['m1', 'm2', 'chieff']

    grid_map = {0: m1grid, 1: m2grid, 2: cfgrid}

    n_dims = len(keep_dims)

    marg_data = marginalize_kde_data(
        data_nd=samples,
        per_point_bw=per_point_bw,
        keep_dims=keep_dims,
        input_transf=input_transf,
        rescale_factors=rescale_factors,
        weights=weights,
        dimension_names=dimension_names
    )

    # We need symmetrization when both dimensions 0 and 1 are present in keep_dims
    # and when they're the first two dimensions in the marginalized space
    if (0 in keep_dims) and (1 in keep_dims):
        # Map original dims 0,1 to their positions in the marginalized space
        keep_dims = list(keep_dims)
        sym_dim_0 = keep_dims.index(0)
        sym_dim_1 = keep_dims.index(1)
        symmetrize_dims = [sym_dim_0, sym_dim_1]
    else:
        symmetrize_dims = None

    # Create evaluation grid
    grids_to_use = [grid_map[dim] for dim in keep_dims]

    if n_dims == 1:
        eval_samples = grids_to_use[0].reshape(-1, 1)
        grid_shape = (len(grids_to_use[0]),)
    elif n_dims == 2:
        mesh_grids = np.meshgrid(grids_to_use[0], grids_to_use[1], indexing='ij')
        eval_samples = np.column_stack([g.ravel() for g in mesh_grids])
        grid_shape = mesh_grids[0].shape
    elif n_dims == 3:
        mesh_grids = np.meshgrid(grids_to_use[0], grids_to_use[1],
                                  grids_to_use[2], indexing='ij')
        eval_samples = np.column_stack([g.ravel() for g in mesh_grids])
        grid_shape = mesh_grids[0].shape
    else:
        raise ValueError(f"Unsupported number of dimensions: {n_dims}")

    # Create KDE with symmetrization if needed
    if 'perpoint_bws' in group:
        train_kde = kde.VariableBwKDEPy(
            marg_data['data'],
            marg_data['weights'],
            input_transf=marg_data['input_transf'],
            stdize=True,
            rescale=marg_data['rescale'],
            bandwidth=marg_data['bandwidth'],
            symmetrize_dims=symmetrize_dims
        )
    else:
        train_kde = akde.AdaptiveBwKDE(
            marg_data['data'],
            marg_data['weights'],
            input_transf=marg_data['input_transf'],
            stdize=True,
            rescale=marg_data['rescale'],
            alpha=alpha,
            symmetrize_dims=symmetrize_dims
        )


    # Evaluate KDE
    eval_kde = train_kde.evaluate_with_transf(eval_samples)

    if n_dims == 1:
        KDE_values = eval_kde.ravel()
    else:
        KDE_values = eval_kde.reshape(grid_shape)

    return {
        'kde_values': KDE_values,
        'grid_shape': grid_shape,
        'eval_samples': eval_samples,
        'marginalized_result': marg_data,
        'kde_object': train_kde
    }


def compute_rate_from_kde(KDE, VT=None, weights_over_VT=None, N=None, vt_weights=True):
    """
    Compute merger rate from KDE using the same weighting scheme.
    
    Parameters
    ---------
    KDE : np.ndarray
        KDE values on grid (can be 1D, 2D, or 3D)
    VT : np.ndarray or None
        Survey volume-time on the same grid (required when vt_weights=False)
    weights_over_VT : np.ndarray or None
        Bootstrap weights divided by VT (if using VT weighting)
    N : int or None
        Number of events (if not using VT weighting)
    vt_weights : bool
        Whether to use VT weighting scheme
        
    Returns
    -------
    Rate : np.ndarray
        Merger rate on the same grid as KDE
        
    Examples
    --------
    # With VT weighting:
    >>> Rate = compute_rate_from_kde(KDE_2d, VT_2d, 
    ...                               weights_over_VT=weights_over_VT, 
    ...                               vt_weights=True)
    
    # Without VT weighting:
    >>> Rate = compute_rate_from_kde(KDE_1d, VT_1d, N=42, vt_weights=False)
    """
    if vt_weights:
        if weights_over_VT is None:
            raise ValueError("weights_over_VT must be provided when vt_weights=True")
        # KDE kernels are weighted by 1/VT
        Rate = weights_over_VT.sum() * KDE
    else:
        if N is None or VT is None::
            raise ValueError("N and VT must both be provided when vt_weights=False")
        Rate = N * KDE / VT

    return Rate

###################################################################
# Get original mean sample points
if opts.samplesx1 and opts.samplesx2 and opts.samplesx3:
    with h5.File(opts.samplesx1, 'r') as f1:
        mean1 = f1['initialdata/original_mean'][...]
    with h5.File(opts.samplesx2, 'r') as f2:
        mean2 = f2['initialdata/original_mean'][...]
    with h5.File(opts.samplesx3, 'r') as f3:
        mean3 = f3['initialdata/original_mean'][...]
    Nev = mean1.size  # Number of detections
else:
    mean1, mean2, mean3 = None, None, None
    Nev = -1  # Sentinel: get nevents from bootstrap

VTdata = h5.File(opts.vt_file, 'r')
m1grid = VTdata['m1vals'][:]
m2grid = VTdata['m2vals'][:]
cfgrid = VTdata['xivals'][:]
VT_3d = VTdata['VT'][...] / 1e9  # change units to Gpc^3
VTdata.close()

if opts.vt_multiplier:  # Scale up for rough estimates if exact VT not available
    VT_3d *= opts.vt_multiplier

hdf = h5.File(opts.iterative_result, 'r')

###### KDE eval 3D grid #########################
XX, YY, ZZ = np.meshgrid(m1grid, m2grid, cfgrid, indexing='ij')
eval_samples = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

######## For 2D plots ##############################
M, CF = np.meshgrid(m1grid, cfgrid, indexing='ij')
M1, M2 = np.meshgrid(m1grid, m2grid, indexing='ij')


threeDgrid = np.array([XX.ravel(), YY.ravel(), ZZ.ravel()]).T

############ Saving data in 3 files #################################
hfintegm1m2 = h5.File(opts.output_tag + "_int_dchieff.hdf5", "w")
hfintegm1chieff = h5.File(opts.output_tag + "_int_dm2.hdf5", "w")
hfintegm2chieff = h5.File(opts.output_tag + "_int_dm1.hdf5", "w")

hfintegm1m2.create_dataset("M1mesh", data=M1)
hfintegm1m2.create_dataset("M2mesh", data=M2)
hfintegm1chieff.create_dataset("Mmesh", data=M)
hfintegm1chieff.create_dataset("CFmesh", data=CF)
hfintegm2chieff.create_dataset("Mmesh", data=M)
hfintegm2chieff.create_dataset("CFmesh", data=CF)

rate_m1m2 = []
ratem1_arr = []
ratem2_arr = []

KDEM1chieff = []
KDEM2chieff = []
RateM1chieff = []
RateM2chieff = []

###############################Iterations and evaluating KDEs/Rate
boots_weighted = False
vt_weights = False  # Flag to control VT weighting

for i in range(opts.end_iter - opts.start_iter):
    it = i + opts.discard + opts.start_iter
    ilabel = i + opts.start_iter
    if it % 5 == 0: print(it)
    iter_name = f'iteration_{it}'
    if iter_name not in hdf:
        print(f"Iteration {it} not found in file.")
        continue

    group = hdf[iter_name]
    if 'bootstrap_weights' in group:
        boots_weighted = True
        weights = group['bootstrap_weights'][:]
        # Allow zero weights as corner case
#       assert min(weights) > 0, "Some bootstrap weights are non-positive!"
        Nboots = weights.sum()  # Number of events in bootstrap

    # Check if VT weighting should be used
    if 'rwvt_vals' in group:
        vt_weights = True
        vt_vals = group['rwvt_vals'][:] / 1e9  # change units to Gpc^3

    samples = group['rwsamples'][:]
    if boots_weighted:
        # Remove samples with zero bootstrap weight as they may have bad behaviour in
        # adaptive KDE (due to extremely small pilot density at the sample location)
        # and have no effect on the KDE, and to reduce compute cost
        samples = samples[weights > 0., :]
        if vt_weights: vt_vals = vt_vals[weights > 0.]
        # Make sure all arrays are the same length
        weights = weights[weights > 0.]

    alpha = group['alpha'][()]
    bwx = group['bwx'][()]
    bwy = group['bwy'][()]
    bwz = group['bwz'][()]
    # Check symmetric dimensions: bwx vs bwy for m1-m2 symmetry
    if not np.isclose(bwx, bwy, rtol=1e-8, atol=0.0):
        print(f"WARNING: bwx != bwy (bwx={bwx}, bwy={bwy}). Setting both to geometric mean.")
        gm = float(np.sqrt(bwx * bwy))  # valid for [0, 1]
        bwx = gm
        bwy = gm
    # Create the KDE with mass symmetry
    m1 = samples[:, 0]  # First column corresponds to m1
    m2 = samples[:, 1]  # Second column corresponds to m2
    cf = samples[:, 2]

    # Determine weights based on vt_weights flag
    if boots_weighted:
        if vt_weights:
            weights = weights / vt_vals
        else:
            pass  # keep weights as they are
    else:
        weights = None

    if opts.integrate_kde == 'marginalized':
    # ========== MARGINALIZED KDE METHOD ==========
        per_point_bandwidth = group['perpoint_bws'][...]

        # Common parameters for all KDE calls
        common_params = {
            'per_point_bw': per_point_bandwidth,
            'input_transf': ('log', 'log', 'none'),
            'rescale_factors': [1/bwx, 1/bwy, 1/bwz],
            'weights': weights,
            'm1grid': m1grid,
            'm2grid': m2grid,
            'cfgrid': cfgrid,
            'alpha': alpha,
            'group': group,
            'dimension_names': ['m1', 'm2', 'chieff']
        }

        # ========== 2D: M1 vs Chieff ==========
        kde_result = create_marginalized_kde(
            samples=samples,
            keep_dims=[0, 2],
            **common_params
        )
        kdeM1chieff = kde_result['kde_values']
        rateM1chieff = compute_rate_from_kde(
            kdeM1chieff, VT_3d,
            weights_over_VT=weights if vt_weights else None,
            N=Nev,
            vt_weights=vt_weights
        )

        # ========== 2D: M2 vs Chieff ==========
        kde_result = create_marginalized_kde(
            samples=samples,
            keep_dims=[1, 2],
            **common_params
        )
        kdeM2chieff = kde_result['kde_values']
        rateM2chieff = compute_rate_from_kde(
            kdeM2chieff, VT_3d,
            weights_over_VT=weights if vt_weights else None,
            N=Nev,
            vt_weights=vt_weights
        )

        # ========== 2D: M1 vs M2 (symmetrization applied in KDE) ==========
        kde_result = create_marginalized_kde(
            samples=samples,
            keep_dims=[0, 1],
            **common_params
        )
        KDEm1m2 = kde_result['kde_values']
        ratem1m2 = compute_rate_from_kde(
            KDEm1m2, VT_3d,
            weights_over_VT=weights if vt_weights else None,
            N=Nev,
            vt_weights=vt_weights
        )
    else:  
        # ========== FULL 3D KDE METHOD ==========
        if 'perpoint_bws' in group:
            per_point_bandwidth = group['perpoint_bws'][...]
            train_kde = kde.VariableBwKDEPy(
                samples,
                weights,
                input_transf=('log', 'log', 'none'),
                stdize=True,
                rescale=[1/bwx, 1/bwy, 1/bwz],
                symmetrize_dims=[0,1],
                bandwidth=per_point_bandwidth
            )
        else:
            train_kde = akde.AdaptiveBwKDE(
                symmetric_samples,
                weights,
                input_transf=('log', 'log', 'none'),
                stdize=True,
                rescale=[1/bwx, 1/bwy, 1/bwz],
                symmetrize_dims=[0,1],
                alpha=alpha
            )

        # Evaluate 3D KDE
        eval_kde3d = train_kde.evaluate_with_transf(eval_samples)
        KDE_3d = eval_kde3d.reshape(XX.shape)

        # Compute 3D rate
        if vt_weights:
            Rate_3d = weights.sum() * KDE_3d
        else:
            N = Nev if Nev > 0 else Nboots
            Rate_3d = N * KDE_3d / VT_3d

        # Calculate marginals by numerical integration
        kdeM1chieff, kdeM2chieff = get_rate_m_chieff2D(m1grid, m2grid, KDE_3d)
        rateM1chieff, rateM2chieff = get_rate_m_chieff2D(m1grid, m2grid, Rate_3d)
        ratem1m2, ratechim1m2, ratechisqm1m2 = integral_wrt_chieff(CF, cfgrid, Rate_3d)
    # Get 1d rates over masses by numerically integrating ratem1m2 over m1>m2
    rateM1, rateM2 = get_rate_m_oneD(m1grid, m2grid, ratem1m2)


    KDEM1chieff.append(kdeM1chieff)
    KDEM2chieff.append(kdeM2chieff)
    RateM1chieff.append(rateM1chieff)
    RateM2chieff.append(rateM2chieff)
    rate_m1m2.append(ratem1m2)
    ratem1_arr.append(rateM1)
    ratem2_arr.append(rateM2)
    hfintegm1m2.create_dataset(f"rate_m1m2_iter{ilabel}", data=ratem1m2)
    hfintegm1chieff.create_dataset(f"rate_m1cf_iter{ilabel}", data=rateM1chieff)
    hfintegm2chieff.create_dataset(f"rate_m2cf_iter{ilabel}", data=rateM2chieff)

    hfintegm1chieff.create_dataset(f"kde_m1cf_iter{ilabel}", data=kdeM1chieff)
    hfintegm2chieff.create_dataset(f"kde_m2cf_iter{ilabel}", data=kdeM2chieff)
    

    if opts.integrate_kde == 'numeric':
        hfintegm1m2.create_dataset(f"rate_chim1m2_iter{ilabel}", data=ratechim1m2)
        hfintegm1m2.create_dataset(f"rate_chisqm1m2_iter{ilabel}", data=ratechisqm1m2)

    hfintegm1m2.create_dataset(f"rate_m1_iter{ilabel}", data=rateM1)
    hfintegm1m2.create_dataset(f"rate_m2_iter{ilabel}", data=rateM2)

    hfintegm1m2.flush()
    hfintegm1chieff.flush()
    hfintegm2chieff.flush()

hfintegm1m2.close()
hfintegm1chieff.close()
hfintegm2chieff.close()

print('Making plots')

rate_m1m2_med = np.percentile(rate_m1m2[:], 50, axis=0)
rate_m1chieff_med = np.percentile(RateM1chieff[:], 50, axis=0)
rate_m2chieff_med = np.percentile(RateM2chieff[:], 50, axis=0)

u_plot.m1m2_contour(mean1, mean2, M1, M2, rate_m1m2_med, timesM=True, itertag=f'{ilabel}', pathplot=opts.pathplot, plot_name='Rate')
u_plot.m_chieff_contour(mean1, mean3, M, CF, rate_m1chieff_med, timesM=True, itertag=f'{ilabel}', pathplot=opts.pathplot, plot_name='Rate', xlabel='m_1')
u_plot.m_chieff_contour(mean1, mean3, M, CF, rate_m2chieff_med, timesM=True, itertag=f'{ilabel}', pathplot=opt.pathplot, plot_name='Rate', xlabel='m_2')
u_plot.oned_rate_mass(m1grid, m2grid, ratem1_arr, ratem2_arr, tag='', pathplot=opts.pathplot)
