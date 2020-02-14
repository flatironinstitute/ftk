#!/usr/bin/env python

import numpy as np
import os

import ftk
import utils


def write_timings(method, oversampling, all_n_trans, all_tm):
    results_dir = utils.get_results_dir()
    filename = 'bench_%s_density%d.csv' % (method, oversampling)
    filename = os.path.join(results_dir, filename)

    with open(filename, 'w') as f:
        for n_trans, tm in zip(all_n_trans, all_tm):
            f.write('%d, %g, %g\n' % (n_trans, tm[0], tm[1]))

def benchmark(oversampling):
    np.random.seed(0)

    print('oversampling = %d' % oversampling)

    T = 2
    n_images = 10

    ftk_eps = 1e-2

    templates = utils.load_images(n_images)

    N = templates.shape[-1]

    rmax = N / 2 * np.pi
    ngridr = 3 * N
    n_psi = 300

    pf_grid = ftk.make_tensor_grid(rmax, ngridr, n_psi)

    Shat = ftk.cartesian_to_pft(templates, T, pf_grid)

    n_tries = 12

    all_tm_bft = []
    all_tm_ftk = []
    all_n_trans = []

    for tr in range(n_tries):
        delta_range = 2 * np.exp(np.log(25/2) * tr / (n_tries - 1));

        print('delta_range = %.3g' % delta_range)

        tr_grid = ftk.make_adaptive_grid(delta_range, T / N, oversampling)

        delta_max = delta_range * T / N
        true_gamma, true_trans = utils.gen_rot_trans(n_images, delta_max)

        Mhat = ftk.rotate_pft(Shat, true_gamma, pf_grid)
        Mhat = ftk.translate_pft(Mhat, true_trans, pf_grid)

        plan = ftk.bft_plan(tr_grid, pf_grid)
        prods_bft, tm_bft = ftk.bft_execute(plan, Mhat, Shat)

        print('BFT, precomp = %.3f s' % tm_bft[0])
        print('BFT, comp = %.3f s' % tm_bft[1])

        n_bessel = int(np.ceil(delta_range))
        plan = ftk.ftk_plan(tr_grid, pf_grid, n_bessel, ftk_eps)
        prods_ftk, tm_ftk = ftk.ftk_execute(plan, Mhat, Shat)


        print('FTK, precomp = %.3f s' % tm_ftk[0])
        print('FTK, comp = %.3f s' % tm_ftk[1])

        err = (np.linalg.norm(prods_bft - prods_ftk)
               / np.linalg.norm(prods_bft))

        n_trans = tr_grid['n_trans']

        print('FTK, relative error = %.3e' % err)

        all_tm_bft.append(tm_bft)
        all_tm_ftk.append(tm_ftk)
        all_n_trans.append(n_trans)

    Nfine = N * oversampling
    Nkeep = 2 * int(np.ceil(oversampling * delta_range))

    plan = ftk.bfr_plan(Nfine, Nkeep, n_psi, pf_grid, T, N)
    prods_bfr, tm_bfr = ftk.bfr_execute(plan, Mhat, Shat)

    print('BFR, precomp = %.3f s' % tm_bfr[0])
    print('BFR, comp = %.3f s' % tm_bfr[1])

    write_timings('bft', oversampling, all_n_trans, all_tm_bft)
    write_timings('ftk', oversampling, all_n_trans, all_tm_ftk)
    write_timings('bfr', oversampling, [0], [tm_bfr])


def main():
    benchmark(2)
    benchmark(4)


if __name__ == '__main__':
    main()
