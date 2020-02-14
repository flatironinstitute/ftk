import os
import matplotlib.pyplot as plt
import numpy as np

import utils


def load_timings(method, oversampling):
    results_dir = utils.get_results_dir()
    filename = 'bench_%s_density%d.csv' % (method, oversampling)
    filename = os.path.join(results_dir, filename)

    tm = np.loadtxt(filename, delimiter=',')

    return tm


def plot_figure6(oversampling):
    bfr_tm = load_timings('bfr', oversampling)
    bft_tm = load_timings('bft', oversampling)
    ftk_tm = load_timings('ftk', oversampling)

    n_trans = bft_tm[:, 0]

    bfr_tm = np.sum(bfr_tm[1:])
    bft_tm = np.sum(bft_tm[:, 1:], axis=1)
    ftk_tm = np.sum(ftk_tm[:, 1:], axis=1)

    blue = [0, 0, 178 / 255]
    orange = [219 / 255, 109 / 255, 0]
    transparent = [0, 0, 0, 0]

    opts = {'markerfacecolor': transparent,
            'linestyle': 'solid',
            'linewidth': 2,
            'markeredgewidth': 2,
            'markersize': 8}

    plt.figure()
    plt.loglog(n_trans, bft_tm, color=blue, marker='o',
               markeredgecolor=blue, **opts)
    plt.loglog(n_trans, bfr_tm * np.ones(len(n_trans)),
               linestyle='dashed', color=[0, 0, 0])
    plt.loglog(n_trans, ftk_tm, color=orange, marker='^',
               markeredgecolor=orange, **opts)
    plt.xlim((n_trans[0], n_trans[-1]))
    plt.ylim((4.64e-1, 4.96e2));
    plt.xlabel('Number of translations N')
    plt.ylabel('Wall clock time (s)')
    plt.legend(['BFT', 'BFR', 'FTK'], loc='lower right')

    if oversampling == 2:
        plt.title('Half-pixel density')
    elif oversampling == 4:
        plt.title('Quarter-pixel density')

    figures_dir = utils.get_figures_dir()

    if oversampling == 2:
        filename = 'figure6a'
    elif oversampling == 4:
        filename = 'figure6b'

    filename = os.path.join(figures_dir, filename + '.eps')

    plt.savefig(filename)


def main():
    plot_figure6(2)
    plot_figure6(4)


if __name__ == '__main__':
    main()
