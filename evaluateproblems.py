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
    stack = list(enumerate(points))

    pointpile = PointPile(x_dim, num_params)

    def pop(Lmin):
        """Find matching sample from points file."""
        # look forward to see if there is an exact match
        # if we do not use the exact matches
        #   this causes a shift in the loglikelihoods
        for i, (idx, next_row) in enumerate(stack):
            row_Lmin = next_row[0]
            L = next_row[1]
            if row_Lmin <= Lmin and L > Lmin:
                idx, row = stack.pop(i)
                return idx, row
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
    plt.savefig(folder + '/simplified_posterior2d.pdf', bbox_inches='tight')
    plt.close()

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
    return np.nanmedian(data)

def compute_problem_width(prob):
    cumprob = prob.cumsum()
    mask_lo = cumprob < 0.05 * prob.sum()
    mask_hi = cumprob > 0.95 * prob.sum()

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

def evaluate_problem(problem_name, folder):
    #print("loading %s ..." % folder)
    #info = json.load(open("%s/info/results.json" % folder))
    sequence, results, livepoint_sequence = getinfo(folder)

    problem_dimensionality = len(results['paramnames'])
    #problem_depth = min(15, info['H'] / problem_dimensionality)
    logvol, p, logl = logVcurve(sequence)
    #print('logvol:', logvol)
    problem_depth_safe = min(10, -np.interp(0.50, p, logvol) / np.log(10) / problem_dimensionality)

    
    prob = results['weighted_samples']['weights']
    logL = results['weighted_samples']['logl']
    # samples = results['weighted_samples']['points']

    problem_width = compute_problem_width(prob)
    problem_width_safe = min(2, problem_width)

    eqsamples = results['samples']
    visualise_problem(folder, problem_name, results, eqsamples)
    problem_gaussianity = compute_gaussian_approximation_loss(eqsamples, prob, logL, problem_dimensionality)
    problem_gaussianity_safe = min(1, problem_gaussianity)

    # logstds = 0.5 * np.log10(np.diag(cov))
    # compute information gain for each parameter
    problem_asymmetry = results['information_gain_bits'].max() / results['information_gain_bits'].min()
    problem_asymmetry = logstds.max() - logstds.min()
    problem_asymmetry_safe = min(6, problem_asymmetry)

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
        #print("  cannot consider all %d clustering combinations ..." % (nmodes_max))
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
        nmodes = max(len(t) for t in thresholds)

    nmodes_safe = min(nmodes, 10)

    #cl = sklearn.cluster.OPTICS(metric='mahalanobis', metric_params=dict(VI=a))
    #print("clustering...")
    #cl.fit(eqsamples[:1000,:])
    #nclusters = len(cl.cluster_hierarchy_)
    #problem_largest_gap = nclusters
    #print("clustering done: %d clusters" % nclusters)

    #Z = scipy.cluster.hierarchy.linkage(eqsamples[:1000,:], method='single', metric='mahalanobis')
    #print(Z[-1], "adds:", Z[-1,-1], Z[-2,-1])
    #print(Z[-1], "adds:", Z[int(Z[-1,-1])-1000,-1], Z[int(Z[-2,-1])-1000,-1])
    #distance_steps = np.log(Z[-30:,2] / Z[-31:-1,2])
    #print(distance_steps, (distance_steps[-1] - distance_steps[:-5].mean()), (distance_steps[:-5]).std())
    #problem_largest_gap = (distance_steps[-1] - distance_steps[:-5].mean())
    #problem_largest_gap = max(0, -problem_largest_gap)

    problem_properties = [
        problem_dimensionality,
        nmodes_safe,
        problem_gaussianity_safe,
        problem_asymmetry_safe,
        problem_width_safe,
        problem_depth_safe,
    ]
    return problem_properties, (logvol, p, logl)

def main():
    #properties_names_short = ["ndim", "multimodality", "non-gaussianity", "asymmetry", "width", "depth"]
    properties_names_short = ['ndim', 'modes', '!gauss', 'asym', 'width', 'depth', 'problem']
    properties_names = ["Dimensionality", "Multimodality", "Non-Gaussianity", "Parameter\nInequality", "Tail Weight", "Depth"]
    bin_separators = [[9, 29], [2, 5], [0.5], [2], [1], [2]]
    bin_members = defaultdict(list)

    fout = open('evaluateproblems.tex', 'w')
    fout.write("""
\begin{tabular}{l|cccc}
%-20s & dim & cost & modes & width & depth & !gauss & asym \\
\hline
\hline
""" % ("name"))
    problem_names = []
    properties_list = []
    LVs = []
    fout.write('\t'.join(properties_names_short) + '\n')
    print('%-20s & %s' % ("name", '\t'.join(properties_names_short)))
    real_problems = set()
    last_field = ""
    for line in open('problems.txt'):
        print(line.rstrip())
        field, problem_name, folder = line.rstrip().split('\t')
        if field != 'toy': continue
        cost = get_cost(folder)
        assert os.path.exists("%s/info/results.json" % folder), folder
        assert os.path.exists("%s/results/points.hdf5" % folder), folder
        p, logVcurve_data = evaluate_problem(problem_name, folder)
        if field != last_field:
            fout.write(r'\multicolumn{8}{l}{%s}' % field)
            fout.write("\n" + r'\hline' + "\n")
        s = '%-20s & %d & %4.0f & %4.1f & %3.1f & %3.2f & %3.2f & %3.2f' % tuple([problem_name] + p[:1] + [cost] + p[1:])
        print(s)
        fout.write(s + '\n')
        problem_names.append(problem_name)
        if field not in ('toy', 'mock'):
            real_problems.add(problem_name)
        properties_list.append(p)

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

        LVs.append((problem_name, logVcurve_data))
    fout.write("""\end{tabular}
""")
    fout.close()

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

    problem_names_unique = sorted(set(problem_names))
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
                    marker='o' if problem_name in real_problems else 's', 
                    ms=12, mew=1, 
                    mfc='None' if problem_name not in real_problems else None, 
                    label=problem_name, 
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
                plt.ylim(None, 104)
            if i == 1:
                plt.yscale('log')
                plt.yticks([1, 2, 4, 10], [1, 2, 4, 10])
            if j == nprops - 1:
                plt.xscale('log')
                plt.xticks([0.5, 1, 5], ['0.5', '1', '5'])
                #plt.xlim(0.3, None)
            if j == 0:
                plt.xlim(0.8, 104)
            if j == 1:
                plt.xscale('log')
                plt.xticks([1, 2, 4, 10], [1, 2, 4, 10])
                #plt.xlim(1, 11)
            #if i == 4:
            yticks, yticklabels = plt.yticks()
            xticks, xticklabels = plt.xticks()
            xlo, xhi = plt.xlim()
            ylo, yhi = plt.ylim()
            #xhi *= 1.2
            #yhi *= 1.3
            plt.vlines(bin_separators[j], ylo, yhi, linestyles=[':']*10, lw=0.2, color='gray', alpha=0.5)
            plt.hlines(bin_separators[i], xlo, xhi, linestyles=[':']*10, lw=0.2, color='gray', alpha=0.5)
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
            plt.legend(loc='best', prop=dict(size=16), bbox_to_anchor=(-2.5, -2.5))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("evaluateproblems.pdf", bbox_inches='tight')
    plt.close()

    print("plotting volcurves")
    plt.figure(figsize=(12, 3.))
    used = {}
    for problem_name, (logvol, p, logl) in LVs:
        # identify point where p=5% and 95%
        #print(problem_name, p)
        volmax, volmid, volmin = np.interp([0.05, 0.50, 0.95], p, logvol)
        color = colors[problem_name]
        m = '-' if problem_name in real_problems else '--'
        if problem_name in used:
            plt.plot(np.exp(logvol), 1 - p, m, color=color)
        else:
            plt.plot(np.exp(logvol), 1 - p, m, label=problem_name, color=color)
        used[problem_name] = True
        plt.plot(np.exp(volmax), [0.95], 'o ', color=color, ms=4)
        plt.plot(np.exp(volmin), [0.05], 'o ', color=color, ms=4)
        plt.plot(np.exp(volmid), [0.50], 'x ', color=color, ms=4, mew=2)

    plt.xscale('log')
    plt.xlim(1e-53, 1)
    plt.xticks([1e-50, 1e-40, 1e-30, 1e-20, 1e-10, 1])
    plt.legend(loc='lower right', ncol=6, bbox_to_anchor=(1.0, 1.04))
    plt.xlabel('Fraction of prior volume', size=20)
    plt.ylabel('Probability enclosed', size=20)
    plt.savefig("evaluateproblems_volcurve.pdf", bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
