from cycler import cycler
import itertools
from datetime import datetime
import numpy as np
import json
import sys, os
import matplotlib.pyplot as plt
from collections import defaultdict
from getdist import MCSamples, plots
from ultranest.netiter import MultiCounter, PointPile, TreeNode, BreadthFirstIterator, combine_results
import h5py
import gzip
import shutil

from joblib import Memory
mem = Memory('.', verbose=False)


def logz_sequence(root, pointpile, nbootstraps=12, random=True, onNode=None, verbose=False, check_insertion_order=True):
    """Run SingleCounter through tree `root`.

    Keeps track of, and returns ``(logz, logzerr, logv, nlive)``.

    Parameters
    ----------
    root: :py:class:`TreeNode`
        Tree
    pointpile: :py:class:`PointPile`
        Point pile
    nbootstraps: int
        Number of independent iterators
    random: bool
        Whether to randomly draw volume estimates
    onNode: function
        Function to call for every node.
        receives current node and the iterator
    verbose: bool
        Whether to show a progress indicator on stderr
    check_insertion_order: bool
        Whether to perform a rolling insertion order rank test

    Returns
    --------
    results: dict
        Run information, see :py:func:`combine_results`
    sequence: dict
        Each entry of the dictionary is results['niter'] long,
        and contains the state of information at that iteration.
        Important keys are:
        Iteration number (niter),
        Volume estimate (logvol), loglikelihood (logl), absolute logarithmic weight (logwt),
        Relative weight (weights), point (samples),
        Number of live points (nlive),
        Evidence estimate (logz) and its uncertainty (logzerr),
        Rank test score (insert_order).

    """
    roots = root.children

    Lmax = -np.inf

    explorer = BreadthFirstIterator(roots)
    # Integrating thing
    main_iterator = MultiCounter(
        nroots=len(roots), nbootstraps=max(1, nbootstraps),
        random=random, check_insertion_order=check_insertion_order)
    main_iterator.Lmax = max(Lmax, max(n.value for n in roots))

    logz = []
    logzerr = []
    nlive = []
    logvol = []
    niter = 0

    livepoint_sequence = []
    saved_nodeids = []
    saved_logl = []
    insert_order = []
    # we go through each live point (regardless of root) by likelihood value
    while True:
        next_node = explorer.next_node()
        if next_node is None:
            break
        rootid, node, (active_nodes, active_rootids, active_values, active_node_ids) = next_node
        # this is the likelihood level we have to improve upon
        Lmin = node.value

        if onNode:
            onNode(node, main_iterator)

        if niter % len(active_values) == 0:
            livepoint_sequence.append((main_iterator.logVolremaining, Lmin, pointpile.us[active_node_ids,:], pointpile.ps[active_node_ids,:]))
        
        logz.append(main_iterator.logZ)
        with np.errstate(invalid='ignore'):
            # first time they are all the same
            logzerr.append(main_iterator.logZerr)
        
        nactive = len(active_values)

        if len(np.unique(active_values)) == nactive and len(node.children) > 0:
            child_insertion_order = (active_values > node.children[0].value).sum()
            insert_order.append(2 * (child_insertion_order + 1.) / nactive)
        else:
            insert_order.append(np.nan)

        nlive.append(nactive)
        logvol.append(main_iterator.logVolremaining)

        niter += 1
        if verbose:
            sys.stderr.write("%d...\r" % niter)

        saved_logl.append(Lmin)
        saved_nodeids.append(node.id)
        # inform iterators (if it is their business) about the arc
        main_iterator.passing_node(rootid, node, active_rootids, active_values)
        explorer.expand_children_of(rootid, node)

    logwt = np.asarray(saved_logl) + np.asarray(main_iterator.logweights)[:,0]
    logvol[-1] = logvol[-2]

    results = combine_results(saved_logl, saved_nodeids, pointpile, main_iterator)
    sequence = dict(
        logz=np.asarray(logz),
        logzerr=np.asarray(logzerr),
        logvol=np.asarray(logvol),
        samples_n=np.asarray(nlive),
        nlive=np.asarray(nlive),
        insert_order=np.asarray(insert_order),
        logwt=logwt,
        niter=niter,
        logl=saved_logl,
        weights=results['weighted_samples']['weights'],
        samples=results['weighted_samples']['points'],
    )

    return sequence, results, livepoint_sequence


def read_file(log_dir, x_dim, num_bootstraps=20, random=True, verbose=False, check_insertion_order=True):
    """
    Read the output HDF5 file of UltraNest.

    Parameters
    ----------
    log_dir: str
        Folder containing results
    x_dim: int
        number of dimensions
    num_bootstraps: int
        number of bootstraps to use for estimating logZ.
    random: bool
        use randomization for volume estimation.
    verbose: bool
        show progress
    check_insertion_order: bool
        whether to perform MWW insertion order test for assessing convergence

    Returns
    ----------
    sequence: dict
        contains arrays storing for each iteration estimates of:

            * logz: log evidence estimate
            * logzerr: log evidence uncertainty estimate
            * logvol: log volume estimate
            * samples_n: number of live points
            * logwt: log weight
            * logl: log likelihood

    final: dict
        same as ReactiveNestedSampler.results and
        ReactiveNestedSampler.run return values

    """
    filepath = os.path.join(log_dir, 'results', 'points.hdf5')
    fileobj = h5py.File(filepath, 'r')
    _, ncols = fileobj['points'].shape
    num_params = ncols - 3 - x_dim

    points = fileobj['points'][:]
    fileobj.close()
    del fileobj
    #stack = list(enumerate(points))
    row_Lmins = points[:,0]
    row_Ls = points[:,1]
    handled = np.zeros(len(points), dtype=bool)
    # first_todo = 0

    pointpile = PointPile(x_dim, num_params)

    def pop(Lmin):
        """Find matching sample from points file."""
        # look forward to see if there is an exact match
        # if we do not use the exact matches
        #   this causes a shift in the loglikelihoods
        mask = np.logical_and(~handled, np.logical_and(row_Lmins <= Lmin, row_Ls > Lmin))
        if not mask.any():
            return None, None
        else:
            idxs, = np.where(mask)
            i = idxs[0]
            handled[i] = True
            return i, points[i]
            
        return None, None

    roots = []
    while True:
        _, row = pop(-np.inf)
        if row is None:
            break
        logl = row[1]
        u = row[3:3 + x_dim]
        v = row[3 + x_dim:3 + x_dim + num_params]
        roots.append(pointpile.make_node(logl, u, v))

    root = TreeNode(id=-1, value=-np.inf, children=roots)

    def onNode(node, main_iterator):
        """Insert (single) child of node if available."""
        while True:
            _, row = pop(node.value)
            if row is None:
                break
            if row is not None:
                logl = row[1]
                u = row[3:3 + x_dim]
                v = row[3 + x_dim:3 + x_dim + num_params]
                child = pointpile.make_node(logl, u, v)
                assert logl > node.value, (logl, node.value)
                main_iterator.Lmax = max(main_iterator.Lmax, logl)
                node.children.append(child)

    return logz_sequence(root, pointpile, nbootstraps=num_bootstraps,
                         random=random, onNode=onNode, verbose=verbose,
                         check_insertion_order=check_insertion_order)


@mem.cache
def getinfo(folder):
    info = json.load(open("%s/info/results.json" % folder))
    ndim = len(info['paramnames'])
    sequence, results, livepoint_sequence = read_file(folder, ndim, verbose=True, random=False, num_bootstraps=4)
    results['paramnames'] = info['paramnames']
    return sequence, results, livepoint_sequence


#@mem.cache
def logVcurve(sequence):
    V, p, logl = np.asarray(sequence['logvol']), np.asarray(sequence['weights']), np.asarray(sequence['logl'])
    i = logl.argsort()
    V_sorted, p_sorted, logl_sorted = V[i], p[i], logl[i]
    c_sorted = p_sorted.cumsum()
    return V_sorted, c_sorted, logl_sorted

def maxof(a, b):
    return np.where(a>b, a, b)

def convex_completion(x, y):
    assert len(x) == len(y), (len(x), len(y))
    ynew = y.copy()
    # iterate from left (lowest x)
    i = 0
    while i < len(x) - 1:
        # draw line to higher values
        ypred = ynew[i] + -1 * (x[i+1:] - x[i])
        # check if it the real curve is above anywhere
        # but skip if the next one is better
        if ynew[i+1] < ypred[0] and np.any(ynew[i+1:] > ypred):
            next_i = np.where(ynew[i+1:] > ypred)[0][0]
            ynew[i+1:i+1+next_i] = maxof(ynew[i+1:i+1+next_i], ypred[:next_i])
        # continue
        i += 1

    return ynew

def visualise_logVcurve(folder, problem_name, logvol, p, logl, logz):
    volmax, volmid, volmin = np.interp([0.025, 0.50, 0.975], p, logvol)
    logl_convex = convex_completion(logvol[::-1], logl[::-1])[::-1]
    logz_simple = np.log(np.trapz(np.exp(logvol), np.exp(logl - logl.max()))) + logl.max()
    logz_convex = np.log(np.trapz(np.exp(logvol), np.exp(logl_convex - logl_convex.max()))) + logl_convex.max()
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.exp(logvol), np.exp(logl - logl.max()))
    plt.plot(np.exp(logvol), np.exp(logl_convex - logl_convex.max()), ':')
    #plt.ylim(1000, + 1)
    #plt.xlim(np.exp(volmin) / 2, np.exp(volmid) * 2)
    plt.xlim(0, np.exp(volmid))
    plt.xlabel('Prior mass')
    plt.ylabel('Likelihood')
    plt.subplot(1, 2, 2)
    plt.title("convex: %.1f%%" % (100 * (1 - np.exp(logz_simple - logz_convex))))
    #plt.title("logz: %.1f" % (logz))
    plt.plot(logvol, logl, label='%.1f' % logz_simple)
    plt.plot(logvol, logl_convex, ':', label='%.1f' % logz_convex)
    #color = l.get_color()
    #plt.plot(volmax, logl.max() + np.log(0.95), 'o ', color=color, ms=4)
    #plt.plot(volmin, logl.max() + np.log(0.05), 'o ', color=color, ms=4)
    #plt.plot(volmid, logl.max() + np.log(0.50), 'x ', color=color, ms=4, mew=2)
    plt.vlines([volmid], logl.max() - 50, logl.max(), colors=['lightgray'])
    plt.fill_between([volmin, volmax], [logl.max() - 50] * 2, [logl.max()] * 2,
        color='lightgray', alpha=0.25)
    plt.ylim(logl.max() - 50, logl.max())
    plt.legend(loc='upper right')
    plt.xlabel('ln(Prior mass)')
    plt.ylabel('ln(Likelihood)')
    plt.subplots_adjust(wspace=0.3)
    plt.savefig(folder + '/logVlogL.pdf')
    plt.close()
    return (1 - np.exp(logz_simple - logz_convex)), volmax, volmid, volmin

def visualise_problem(folder, problem_name, info, eqsamples):
    if os.path.exists(folder + '/simplified_posterior2d.pdf'):
        return

    transformed_samples = np.copy(eqsamples)
    paramnames = list(info['paramnames'])
    if len(paramnames) > 15:
        return
    print(paramnames, transformed_samples.shape)
    for i in range(len(paramnames)):
        if problem_name == 'eggbox':
            continue
        elif eqsamples[:,i].min() > 0 and eqsamples[:,i].max() / eqsamples[:,i].min() > 1000 and eqsamples[:,i].max() > 30:
            transformed_samples[:,i] = np.log10(eqsamples[:,i])
            print("  applying log(%s)" % paramnames[i])
            paramnames[i] = 'log(%s)' % paramnames[i]
        elif paramnames[i].startswith('log'):
            paramnames[i] = 'log(%s)' % paramnames[i][3:]
        del i

    default_i, default_j = 0, 1
    default_smooth_scale_2D = 1.1
    if problem_name == 'testmultisine' and len(paramnames) > 3:
        i, j = 2, 3
        smooth_scale_2D = default_smooth_scale_2D
    elif problem_name == 'asymgauss':
        i, j = 0, -1
        smooth_scale_2D = default_smooth_scale_2D
    elif problem_name == 'xrayspectrum10' or problem_name == 'bixrayspectrum':
        i, j = 0, 2
        smooth_scale_2D = 2.0
    elif problem_name == 'ligo':
        i, j = default_i, default_j
        smooth_scale_2D = 3.0
    elif problem_name == 'distbeta':
        i, j = default_i, default_j
        smooth_scale_2D = 3.0
    elif problem_name.startswith('exo-rv') and len(paramnames) > 3:
        i, j = 0, 3
        smooth_scale_2D = default_smooth_scale_2D
    elif problem_name == 'cosmology':
        i, j = 1, 4
        smooth_scale_2D = 3.0
    elif problem_name == 'rosen':
        i, j = default_i, default_j
        smooth_scale_2D = 2.0
    else:
        print("plotting first two parameters for:", problem_name)
        i, j = default_i, default_j
        smooth_scale_2D = default_smooth_scale_2D

    samples_g = MCSamples(samples=transformed_samples,
                          names=paramnames,
                          label=problem_name,
                          settings=dict(smooth_scale_2D=smooth_scale_2D),
                          sampler='nested')

    mcsamples = [samples_g]

    try:
        plt.figure()
        plt.title(problem_name)
        g = plots.get_subplot_plotter(width_inch=8)
        g.settings.num_plot_contours = 3
        g.triangle_plot(mcsamples, filled=False, contour_colors=plt.cm.Set1.colors)
        print("plotting to %s/simplified_posterior.pdf" % folder)
        plt.savefig(folder + '/simplified_posterior.pdf', bbox_inches='tight')
        plt.close()
        del g
    except Exception as e:
        print(e)

    #plt.figure(figsize=(3,4))
    g = plots.get_single_plotter(width_inch=4, ratio=1)
    plt.title(problem_name)
    g.settings.num_plot_contours = 3
    g.plot_2d(mcsamples, paramnames[i], paramnames[j], filled=False,
        shaded=True, contour_colors=plt.cm.Set1.colors)
    plt.xlim(transformed_samples[:,i].min(), transformed_samples[:,i].max())
    plt.ylim(transformed_samples[:,j].min(), transformed_samples[:,j].max())
    #plt.scatter(transformed_samples[:,i], transformed_samples[:,j], marker='.')
    #plt.xlabel(paramnames[i])
    #plt.ylabel(paramnames[j])
    print("plotting to %s/simplified_posterior2d.pdf" % folder)
    plt.savefig(folder + '/simplified_posterior2d.pdf')
    plt.close()


def simplify_problem_name(problem_name):
    if problem_name.startswith('spikeslab'):
        return 'spike+slab'
    if problem_name.split('-')[-1].endswith('d'):
        try:
            int(problem_name.split('-')[-1][:-1])
            return '-'.join(problem_name.split('-')[:-1])
        except Exception:
            pass
    try:
        int(problem_name.split('-')[-1])
        return '-'.join(problem_name.split('-')[:-1])
    except Exception:
        pass
    return problem_name

def get_cost(folder):
    data = []
    last_ncalls = None
    last_time = None
    # for line in open("%s/debug.log" % folder):
    for line in gzip.open("%s/debug.log.gz" % folder, 'rt'):
        if '[DEBUG] iteration=' in line:
            parts = line.split()
            time = datetime.strptime(parts[0], "%H:%M:%S")
            ncalls = int(parts[4].replace('ncalls=', '').strip(','))
            # if time did not advance, skip ahead
            if last_time is not None and time == last_time:
                # print('skipping', line.rstrip())
                continue
            
            if last_ncalls is not None and ncalls != last_ncalls:
                data.append( ((time - last_time) * 1000).total_seconds() / (ncalls - last_ncalls) )
                # print( (time - last_time), 'for', (ncalls - last_ncalls), 'calls', data[-1])

            last_ncalls = ncalls
            last_time = time

    # print("conclusion:", np.nanmedian(data))
    if len(data) == 0 and folder == 'systematiclogs/cmb-cosmology/ultranest-safe/':
        return 1000 # about one per second, estimated from log file by eye

    return np.nanmedian(data[-1000:])

def compute_problem_width_orig(prob):
    cumprob = prob.cumsum()
    mask_lo = cumprob < 0.025 * prob.sum()
    mask_hi = cumprob > 0.975 * prob.sum()

    prob_lo = prob[mask_lo][-1]
    prob_hi = prob[mask_hi][0]

    return np.log10(prob_hi) - np.log10(prob_lo)

def compute_gaussian_approximation_loss(eqsamples, prob, logL, problem_dimensionality):
    cov = np.cov(eqsamples, rowvar=0)
    mean = np.mean(eqsamples, axis=0)

    # compute likelihood given that gaussian parameters
    a = np.linalg.inv(cov)
    delta = eqsamples - mean
    mahalanobis_distances = np.einsum('...i, ...i', np.tensordot(delta, a, axes=1), delta)
    logLgauss = -0.5 * mahalanobis_distances # - 0.5 * np.sum(np.log(np.pi * np.diag(cov)))

    # compute KL divergence:
    #    reference distribution is equally sampled. so prob = 1/N
    #    new distribution has probability logLgauss
    surprise = np.sum(prob * (logL - logL.max() - logLgauss))

    return np.abs(surprise) / problem_dimensionality

def count_posterior_modes(problem_name, eqsamples, problem_dimensionality):
    """Count modes.

    Makes a histogram of the posterior samples in each parameter,
    makes a cut at 20% of the maximum posterior.
    If there is a gap below that, creates a split in that parameter.

    Finally, iterates through all split combinations and counts
    each "mode" if it contains posterior samples.
    """
    #nmodes = 1
    # count the number of clusters
    # 1) project each axis and find cuts
    thresholds = []
    for i in range(problem_dimensionality):
        threshold_here = []
        H, edges = np.histogram(eqsamples[:,i], bins=20)
        threshold = H.max() / 5
        assert len(edges) == len(H) + 1, (len(edges), len(H))
        # trim ends
        while H[0] < threshold:
            H = H[1:]
            edges = [edges[0]] + list(edges[2:])
            assert len(edges) == len(H) + 1, (len(edges), len(H))
        while H[-1] < threshold:
            H = H[:-1]
            edges = list(edges[:-2]) + [edges[-1]]
            assert len(edges) == len(H) + 1, (len(edges), len(H))

        edges_lo = edges[:-1]
        edges_hi = edges[1:]
        
        #print(i, H)
        for k, g in itertools.groupby(zip(H > threshold, edges_lo, edges_hi), key=lambda e: e[0]):
            if not k: continue
            los = []
            his = []
            for _, lo, hi in g:
                los.append(lo)
                his.append(hi)
            threshold_here.append([i, min(los), max(his)])
        #nmodes = max(nmodes, len([k for k, _ in itertools.groupby(H > threshold) if k]))
        #assert len(threshold_here) == nmodes
        #if len(threshold_here) > 1:
        #   print("  found %d modes in dimension %d" % (len(threshold_here), i))
        thresholds.append(threshold_here)

    # 2) find all combinations of cuts and deduplicate
    cluster_assignment = -np.ones(len(eqsamples), dtype=int)
    nmodes_max = np.product([len(t) for t in thresholds])
    # deduplicate:
    if nmodes_max > 10000:
        print("  cannot consider all %d clustering combinations ..." % (nmodes_max))
        nmodes = nmodes_max
    elif sum(len(t) > 1 for t in thresholds) > 1:
        #print("  considering %d clustering combinations ..." % (nmodes_max))
        for clusterid, combination in enumerate(itertools.product(*thresholds)):
            # select samples that fulfill all criteria
            selection_mask = np.ones(len(eqsamples), dtype=bool)
            for i, lo, hi in combination:
                mask = np.logical_and(eqsamples[:,i] >= lo, eqsamples[:,i] <= hi)
                selection_mask = np.logical_and(selection_mask, mask)

            if np.all(cluster_assignment[selection_mask] == -1):
                #print("new cluster:", selection_mask.sum(), combination)
                cluster_assignment[selection_mask] = clusterid
            else:
                # this already belongs to another cluster
                # merge all the samples clusters
                # get clusters to merge
                clusters = np.unique(cluster_assignment[selection_mask])
                #print("merging:", clusters)
                # target cluster is the smallest one
                clusterid_target = clusters.min()
                # handle unassigned ones
                cluster_assignment[cluster_assignment[selection_mask] == -1] == clusterid_target
                # handle assigned ones
                for clusterid in clusters:
                    cluster_assignment[cluster_assignment == clusterid] == clusterid_target

        clusters = np.unique(cluster_assignment)
        nmodes = (clusters > -1).sum()
    else:
        # at most one axis has splits
        nmodes = max(len(t) for t in thresholds)
    if problem_name.startswith('beta') and problem_dimensionality == 30:
        nmodes = 1024
    return nmodes


def trim_properties(problem_dimensionality,
        problem_depth,
        problem_width,
        nmodes,
        problem_asymmetry,
        problem_gaussianity,
        problem_convexity):

    problem_depth_safe = min(3, problem_depth)
    problem_width_safe = max(0, min(7, problem_width))
    nmodes_safe = min(nmodes, 10)
    problem_asymmetry_safe = min(100, problem_asymmetry)
    problem_gaussianity_safe = min(0.5, problem_gaussianity)
    problem_convexity_safe = max(1e-3, problem_convexity)
    return [
        problem_dimensionality,
        problem_depth_safe,
        problem_width_safe,
        nmodes_safe,
        problem_asymmetry_safe,
        problem_gaussianity_safe,
        problem_convexity_safe,
    ]


def evaluate_problem(problem_name, folder):
    sequence, results, livepoint_sequence = getinfo(folder)
    
    u = results['weighted_samples']['upoints']
    w = results['weighted_samples']['weights']
    IG = np.zeros(len(results['posterior']['information_gain_bits']))
    for i in range(len(IG)):
        H, _ = np.histogram(u[:,i], weights=w, density=True, bins=np.linspace(0, 1, 1000))
        H[H < 1e-10] = 1e-10
        Href = H * 0 + 1.
        assert np.allclose(Href.mean(), 1)
        IG[i] = (np.log2(Href / H) * Href).mean()

    problem_dimensionality = len(results['paramnames'])
    #problem_depth = min(15, info['H'] / problem_dimensionality)
    logvol, p, logl = logVcurve(sequence)
    problem_convexity, volmax, volmid, volmin = visualise_logVcurve(folder, problem_name, logvol, p, logl, results['logz'])

    problem_depth = -volmid / np.log(10) / problem_dimensionality
    # problem_width = compute_problem_width(logvol, p, logl)
    problem_width = abs(volmax - volmin) / np.log(10) - np.log10(problem_dimensionality)
    
    prob = results['weighted_samples']['weights']
    logL = results['weighted_samples']['logl']
    # samples = results['weighted_samples']['points']


    eqsamples = results['samples']
    visualise_problem(folder, problem_name, results, eqsamples)
    problem_gaussianity = compute_gaussian_approximation_loss(eqsamples, prob, logL, problem_dimensionality)

    # logstds = 0.5 * np.log10(np.diag(cov))
    # compute information gain for each parameter
    # problem_asymmetry = logstds.max() - logstds.min()
    problem_asymmetry = max(IG.max(), 1) / max(IG.min(), 1)
    print("asymmetry:",  problem_asymmetry,  IG.max(), IG.min(), IG)

    nmodes = count_posterior_modes(problem_name, eqsamples, problem_dimensionality)
    
    problem_properties = [
        problem_dimensionality,
        problem_depth,
        problem_width,
        nmodes,
        problem_asymmetry,
        problem_gaussianity,
        problem_convexity,
    ]
    return problem_properties, (logvol, p, logl)

def main(filename):
    #properties_names_short = ["ndim", "multimodality", "non-gaussianity", "asymmetry", "width", "depth"]
    properties_names_short = ['dim', 'depth', 'width', 'modes', 'asym', '!gauss', 'phase']
    properties_names = ["Dimensionality", "Depth", "Width", "Modes", "Inequality", "Non-Gaussianity", 'Transition']

    bin_separators = [[9, 29], [2], [4], [2, 5], [5], [0.2], [0.02]]
    bin_members = defaultdict(list)

    fout = open('evaluateproblems.tex.tmp', 'w')
    fout.write("""
\\begin{tabular}{ll|%s}
%-20s & %-20s & %s \\\\
\\hline
\\hline
""" % (
    'rr' + 'c' * (len(properties_names_short) - 1), "field", "name", 
    ' & '.join(properties_names_short[:1] + ['cost'] + properties_names_short[1:])))
    problem_names = []
    properties_list = []
    LVs = []
    print('%-20s & %s' % ("name", '\t'.join(properties_names_short[:1] + ['cost'] + properties_names_short[1:])))
    real_problems = set()
    mock_problems = set()
    last_field = ""

    plt.rcParams['axes.prop_cycle'] = cycler('color', [
        '#008fd5', '#fc4f30', '#e5ae38', '#6d904f', '#8b8b8b', '#810f7c', '#111111',
        'tab:brown',
        'tab:pink',
        'tab:olive',
        'tab:gray',
        'tab:cyan',
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
    ])
    for line in open(filename):
        if line.startswith('#'):
            continue
        print(line.rstrip())
        field, problem_name_orig, folder = line.rstrip().split('\t')
        #if field != 'mock': continue
        problem_name = simplify_problem_name(problem_name_orig)
        #if problem_name == 'spike+slab':
        #    continue
        cost = get_cost(folder)
        assert os.path.exists("%s/info/results.json" % folder), folder
        assert os.path.exists("%s/results/points.hdf5" % folder), folder
        p_unsafe, logVcurve_data = evaluate_problem(problem_name, folder)
        p = trim_properties(*p_unsafe)
        if field != last_field and field in ('toy', 'mock'):
            fout.write(r'\hline' + "\n")
        #    fout.write(r'\multicolumn{9}{l}{%s} \\' % field)
        #    fout.write("\n" + r'\hline' + "\n")
        #    last_field = field
        s = '%-20s & %-20s & %-3d & %-4d & %-4.2f & %-2.1f & %4d & %-3.2f & %-3.2f & %-2.3f' % tuple(
            [field if field != last_field else '', problem_name_orig] + p_unsafe[:1] + [cost] + p_unsafe[1:])
        
        last_field = field
        print(s)
        fout.write(s + ' \\\\\n')
        fout.flush()
        problem_names.append(problem_name)
        if field not in ('toy', 'mock'):
            real_problems.add(problem_name)
        elif field == 'mock':
            mock_problems.add(problem_name)
        
        properties_list.append(p)

        """
        index = 0
        for prop, seps in zip(p, bin_separators):
            index_here = sum(prop >= s for s in seps)
            # print(index_here, "from", prop, seps)
            index *= (len(seps) + 1)
            index += index_here
            del seps

        if len(bin_members[index]) > 0:
            print("%s redundant with %s: index %d" % (folder, bin_members[index][0], index))
        bin_members[index].append(folder)
        del index
        """

        LVs.append((problem_name, logVcurve_data))
    fout.write("""\end{tabular}
""")
    fout.close()
    shutil.move('evaluateproblems.tex.tmp', 'evaluateproblems.tex')

    print("analysing what types of problems are missing...")
    out = open('evaluate_missing.txt', 'w')
    for p in itertools.product(*[range(len(seps) + 1) for seps in bin_separators]):
        index = 0
        name = []
        for index_here, seps, pname in zip(p, bin_separators, properties_names_short):
            index = index * (len(seps) + 1) + index_here
            if index_here == 0:
                name.append("%6s<%s" % (pname, seps[0]))
            elif index_here == len(seps):
                name.append("%6s>%s" % (pname, seps[-1]))
            else:
                name.append("%6s=%s-%s" % (seps[index_here - 1], pname, seps[index_here]))
        out.write("%d: %10s\n" % (len(bin_members[index]), ' '.join(name)))
        del p

    problem_names_unique = []
    for p in problem_names:
        if p not in problem_names_unique:
            problem_names_unique.append(p)
    properties = np.array(properties_list)
    nprops = properties.shape[1]

    colors = {}

    print("plotting grid...")
    plt.figure(figsize=(20, 20))
    for i in range(nprops):
        for j in range(i + 1, nprops):
            plt.subplot(nprops, nprops, i * nprops + j + 1)
            for problem_name in problem_names_unique:
                mask = [p == problem_name for p in problem_names]
                r, = plt.plot(properties[mask,j], properties[mask,i],
                    marker='o' if problem_name in real_problems else ('s' if problem_name in mock_problems else 'x'),
                    ms=12 if problem_name != 'spike+slab' else 4, mew=1, 
                    mfc='None' if problem_name not in real_problems else None, 
                    label=problem_name, 
                    alpha=0.5 if problem_name == 'spike+slab' else 1,
                    ls=' ')
                colors[problem_name] = r.get_color()
            if j == i + 1:
                if j == nprops - 1:
                    plt.xlabel(properties_names[j], size=16)
                else:
                    plt.xlabel("\n\n\n" + properties_names[j], size=16)
                if i == 0:
                    plt.ylabel(properties_names[i], size=16)
            if i == 0:
                plt.yscale('log')
                plt.yticks([1, 5, 10, 30, 100], [1, 5, 10, 30, 100])
                #plt.ylim(1, 104)
            # plt.text(0.98, 0.98, 'i=%d,j=%d' % (i,j), va='top', ha='right', transform=plt.gca().transAxes)
            if i == 3:
                plt.yscale('log')
                plt.yticks([1, 2, 4, 10], [1, 2, 4, 10])
            if j == 3:
                plt.xscale('log')
                plt.xticks([1, 2, 4, 10], [1, 2, 4, 10])
            if j == 4:
                plt.xscale('log')
                plt.xticks([1, 2, 4, 10, 40, 100], [1, 2, 4, 10, 40, 100])
            if i == 4:
                plt.yscale('log')
                plt.yticks([1, 2, 4, 10, 40, 100], [1, 2, 4, 10, 40, 100])
            #if j == 5:
            #    plt.xscale('log')
            #if i == 5:
            #    plt.yscale('log')
            if j == 6:
                plt.xscale('log')
            #if j == nprops - 2:
            #    plt.xscale('log')
            #    plt.xticks([0.5, 1, 5], ['0.5', '1', '5'] if i == j else [''] * 3)
            #    #plt.xlim(1, 11)
            yticks, yticklabels = plt.yticks()
            xticks, xticklabels = plt.xticks()
            xlo, xhi = plt.xlim()
            ylo, yhi = plt.ylim()
            #xhi *= 1.2
            #yhi *= 1.3
            plt.vlines(bin_separators[j], ylo, yhi, lw=0.2, color='gray', alpha=0.5)
            plt.hlines(bin_separators[i], xlo, xhi, lw=0.2, color='gray', alpha=0.5)
            if j != i + 1:
                if yticks[-1] / yhi > 0.95:
                    plt.yticks(yticks[:-1], ['' for t in yticks[:-1]])
                if xticks[-1] / xhi > 0.95:
                    plt.xticks(xticks[:-2], ['' for t in xticks[:-2]])
            elif j < nprops - 1:
                if yticks[-1] / yhi > 0.95:
                    plt.yticks(yticks[:-1])
                if xticks[-1] / xhi > 0.8:
                    plt.xticks(xticks[:-2])
            else:
                print("not modifying ticks for %d/%d" % (i, j))
            plt.xlim(xlo, xhi)
            plt.ylim(ylo, yhi)

        if i == 0:
            plt.legend(loc='upper left', prop=dict(size=16), bbox_to_anchor=(-5.5, -1.0), numpoints=1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("evaluateproblems.pdf", bbox_inches='tight')
    plt.close()

    print("plotting volcurves")
    plt.figure(figsize=(10, 3.))
    used = {}
    for problem_name, (lnvol, p, logl) in LVs:
        # identify point where p=5% and 95%
        p[0] = 0
        # print(problem_name, p[0], logvol[0])
        logvol = lnvol / np.log(10)
        #volmax, volmid, volmin = np.interp([0.05, 0.50, 0.95], p, logvol)
        color = colors[problem_name]
        m = '-' if problem_name in real_problems else ('--' if problem_name in mock_problems else ':')
        if problem_name in used:
            plt.plot(-logvol, 1 - p, m, color=color, alpha=0.2 if problem_name == 'spike+slab' else 1)
        else:
            plt.plot(-logvol, 1 - p, m, label=problem_name, color=color, alpha=0.2 if problem_name == 'spike+slab' else 1)
        used[problem_name] = True
        #plt.plot(-volmax, [0.95], 'o ', color=color, ms=4)
        #plt.plot(-volmin, [0.05], 'o ', color=color, ms=4)
        #plt.plot(-volmid, [0.50], 'x ', color=color, ms=4, mew=2)

    plt.xscale('log')
    plt.xlim(0.5, 260)
    plt.xticks([100, 30, 10, 3, 1], ['$10^{-100}$', '$10^{-30}$', '$10^{-10}$', '$10^{-3}$', '$10^{-1}$'])
    #plt.gca().xaxis.grid(False, which='minor')
    #plt.gca().xaxis.grid(False, which='major')
    plt.legend(loc='lower right', ncol=6, bbox_to_anchor=(1.0, 1.04))
    plt.xlabel('Prior mass')
    plt.ylabel('Posterior enclosed')
    plt.savefig("evaluateproblems_volcurve.pdf", bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main(sys.argv[1])
