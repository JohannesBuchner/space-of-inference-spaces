import argparse
import numpy as np
from numpy import exp, log1p, log
import matplotlib.pyplot as plt

def main(args):
    paramnames = ['intercept', 'slope']

    log_dir = '%s' % (args.log_dir)
    
    x = [0.50,	0.75,	1.00,	1.25,	1.50,	1.75,	1.75,	2.00,	2.25,	2.50,	2.75,	3.00,	3.25,	3.50,	4.00,	4.25,	4.50,	4.75,	5.00,	5.50]
    y = [0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	1,	0,	1,	1,	1,	1,	1,	1]
    x = np.array(x)
    y = np.array(y) > 0

    def loglike(params):
        x0, slope = params
        p = 1 / (1 + exp(-((x - x0)*slope)))
        m = 1 - p
        logl = np.where(y, log(p + 1e-300), log(m + 1e-300)).sum()
        return logl

    def transform(x):
        z = x.copy()
        z[0] = 10**(x[0] * 3 - 1)
        z[1] = x[1] * 10 - 5
        return z

    from ultranest import ReactiveNestedSampler
    sampler = ReactiveNestedSampler(paramnames, loglike, transform=transform, 
        log_dir=log_dir, resume=True)
    sampler.run(min_num_live_points=args.num_live_points)
    sampler.print_results()
    sampler.plot()

    from ultranest.plot import PredictionBand
    plt.scatter(x, y)
    x_l = np.linspace(0.1, 10, 1000)
    band = PredictionBand(x_l)
    for x0, slope in sampler.results['samples']:
        y_pred = 1 / (1 + exp(-((x_l - x0)*slope)))
        band.add(y_pred)
    band.line(color='gray')
    band.shade(color='gray', alpha=0.2)
    plt.savefig(log_dir + '/output.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_live_points", type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='logs/logistic')
    parser.add_argument('--reactive', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
