#!/usr/bin/env python

import numpy as np

from time import time

import pyfftw

from numpy.fft import fft, ifft, fftshift, ifftshift, fft2, ifft2

from scipy.special import jv as besselj

import finufft

def translations_brute_force(Shathat, Mhat, cmul_trans):
    # Shathat: (q, te, k)
    # Mhat: (im, k × γ)
    # cmul_trans: (tr, k × γ)

    n_trans = cmul_trans.shape[-2]
    n_images = Mhat.shape[-2]

    Shathat = Shathat.transpose((2, 0, 1))
    # Shathat: (q, te, k)

    n_templates = Shathat.shape[-2]
    ngridr = Shathat.shape[-1]
    n_gamma = Shathat.shape[-3]

    Mhat = Mhat.reshape((n_images, ngridr, n_gamma))
    cmul_trans = cmul_trans.reshape((n_trans, ngridr, n_gamma))
    # Mhat: (im, k, γ)
    # cmul_trans: (tr, k, γ)

    Mhat = Mhat[:, np.newaxis, :, :]
    cmul_trans = cmul_trans[np.newaxis, :, :, :]
    # Mhat: (im, 1, k, γ)
    # cmul_trans: (1, tr, k, γ)

    Mhat = Mhat.transpose((3, 2, 0, 1)).copy()
    cmul_trans = cmul_trans.transpose((3, 2, 0, 1)).copy()
    # Mhat: (γ, k, im, 1)
    # cmul_trans: (γ, k, 1, tr)

    Mhat_trans = pyfftw.empty_aligned((n_gamma, ngridr, n_images, n_trans),
        dtype='complex128')
    # Mhat_trans: (γ, k, im × tr)

    plan = pyfftw.FFTW(Mhat_trans, Mhat_trans, axes=(0,),
        direction='FFTW_FORWARD', flags=('FFTW_ESTIMATE',), threads=12)

    tmr_start = time()

    np.multiply(Mhat, cmul_trans, out=Mhat_trans)

    plan()
    Mhathat_trans = Mhat_trans.reshape((n_gamma, ngridr, n_images * n_trans))
    # Mhathat_trans: (q, k, im × tr)

    ptm = time() - tmr_start

    tmr_start = time()
    c_n2 = np.zeros((n_gamma, n_templates, n_images*n_trans),
                    dtype=np.complex128)
    # c_n2: (q, te, im × tr)

    for k1 in range(n_gamma):
        k1p = (k1 + n_gamma // 2) % n_gamma
        c_n2[k1, :, :] = np.matmul(np.conj(Shathat[k1p, :, :]), Mhathat_trans[k1, :, :])
    c_n2 = 2 * np.pi * c_n2
    c_n2 = ifft(c_n2, axis=0)
    # c_n2: (γ, te, im × tr)

    c_n2 = c_n2.reshape((n_gamma, n_templates, n_images, n_trans))
    c_n2 = np.real(c_n2)
    # c_n2: (γ, te, im, tr)

    tm = time() - tmr_start

    return c_n2, ptm, tm


def translations_brute_force_batch(Shathat, Mhat, pf_grid, tr_grid, n_psi,
        n_batch_im=None, n_batch_trans=500):
    n_templates = Shathat.shape[0]
    n_images = Mhat.shape[0]
    trans = tr_grid['trans']
    n_trans = tr_grid['n_trans']

    if n_batch_im is None:
        n_batch_im = n_images

    n_batch_trans = min(n_batch_trans, n_trans)

    zprods1 = np.zeros((n_psi, n_templates, n_images, n_trans))
    # zprods1: (γ, te, im, tr)

    tm1 = 0
    precomp1 = 0

    for cn in range(0, n_images, n_batch_im):
        idx_im = range(cn, min(cn + n_batch_im, n_images))

        for ttt in range(0, n_trans, n_batch_trans):
            idx_trans = range(ttt, min(ttt + n_batch_trans, n_trans))

            cmul_trans = pft_phase_shift(-trans[idx_trans, :], pf_grid)
            # cmul_trans: (tr, k × γ)

            tmp, ptm, tm = translations_brute_force(
                Shathat, Mhat[idx_im, :], cmul_trans)

            zprods1[np.ix_(range(n_psi),
                           range(n_templates),
                           idx_im,
                           idx_trans)] = tmp
            precomp1 += ptm
            tm1 += tm

    zprods1 = zprods1.transpose((2, 1, 0, 3))

    return zprods1, precomp1, tm1


def svd_decomposition_alignment(SSS, Mhat, n_bessel, all_rnks, BigMul_left):
    ngridr = SSS.shape[-1]
    n_templates = SSS.shape[-2]
    n_gamma = SSS.shape[-3]
    n_images = Mhat.shape[-2]
    n_trans = BigMul_left.shape[-1]

    tmr_start = time()
    Mhathat = Mhat.reshape((n_images, ngridr, n_gamma))
    Mhathat = fftshift(fft(Mhathat, axis=-1), axes=-1) / n_gamma

    MMM = np.zeros((n_images, 2 * n_bessel + 1, ngridr, n_gamma),
                   dtype=np.complex128)

    for im in range(n_images):
        for qp in range(-n_bessel, n_bessel + 1):
            tmp = Mhathat[im, :, :]
            MMM[im, qp + n_bessel, :, :] = np.roll(tmp, -qp, axis=-1)

    MMM = MMM.transpose((1, 3, 2, 0)).copy()
    precomp2 = time() - tmr_start

    tmr_start = time()
    BigMul_right = np.zeros((sum(all_rnks), n_gamma, n_templates, n_images),
                            dtype=np.complex128)
    for qp in range(-n_bessel, n_bessel + 1):
        rnk = all_rnks[qp + n_bessel]
        ofst = sum(all_rnks[:qp + n_bessel])
        for ll in range(rnk):
            for q in range(n_gamma):
                tmp = np.matmul(SSS[ofst + ll, q, :, :],
                                MMM[qp + n_bessel, q, :, :])
                BigMul_right[ofst + ll, q, :, :] = tmp

    BigMul_right = BigMul_right.transpose((3, 2, 1, 0)).copy()

    c_n = np.zeros((n_images, n_templates, n_gamma, n_trans),
                   dtype=np.complex128)
    for im in range(n_images):
        for tt in range(n_templates):
            c_n[im, tt, :, :] = np.matmul(BigMul_right[im, tt, :, :],
                                          BigMul_left)
    c_n = 2 * np.pi * c_n

    zprods = ifft(ifftshift(c_n, axes=-2), axis=-2) * n_gamma
    tm2 = time() - tmr_start

    return zprods, precomp2, tm2


def cartesian_to_pft(templates, T, pf_grid):
    xnodesr = pf_grid['xnodesr']
    n_psi = pf_grid['n_psi']

    ngridr = xnodesr.shape[0]

    n_templates = templates.shape[0]
    N = templates.shape[1]

    dx = T / N
    dy = T / N

    wx = pf_grid['wx']
    wy = pf_grid['wy']

    Shat = np.zeros((n_templates, ngridr * n_psi), dtype=np.complex128)

    upsampfac = 1.25

    fcc = np.empty(len(wx), dtype=np.complex128)

    for k in range(n_templates):
        template = templates[k, :, :]

        # Need to force Fortran ordering because that's what the FINUFFT
        # interface expects.
        gg = np.asarray(template,dtype=np.complex128)
        #gg = np.asfortranarray(template.transpose((1, 0)))

        isign = -1
        eps = 1e-6

        # Note: Crashes if gg is a 1D vector (raveled). Why?
        finufft.nufft2d2(wx * dx, wy * dy, gg,
                           isign=isign, eps=eps, out=fcc, upsampfac=upsampfac)
        Shat[k, :] = fcc

    return Shat


def pft_to_cartesian(Shat, T, N, pf_grid):
    xnodesr = pf_grid['xnodesr']
    n_psi = pf_grid['n_psi']
    quad_wts = pf_grid['quad_wts']

    ngridr = xnodesr.shape[0]

    n_templates = Shat.shape[0]

    dx = T / N
    dy = T / N

    wx = pf_grid['wx']
    wy = pf_grid['wy']

    templates1 = np.zeros((n_templates, N, N))

    gxx = np.empty((N, N), dtype=np.complex128)

    upsampfac = 1.25

    for k in range(n_templates):
        fcc1 = Shat[k, :] * quad_wts

        isign = 1
        eps = 1e-6

        finufft.nufft2d1(wx * dx, wy * dy, fcc1, (N,N), isign=isign, eps=eps,
                         out=gxx,upsampfac=upsampfac)

        gxx = gxx*dx*dy/(4*np.pi**2)

        templates1[k, :, :] = np.real(gxx.transpose((1, 0)))

    return templates1


def rotate_pft(fcc, rgamma, pf_grid):
    xnodesr = pf_grid['xnodesr']
    n_psi = pf_grid['n_psi']

    ngridr = xnodesr.shape[0]

    ngridc = n_psi * np.ones(ngridr, dtype=np.int32)

    fcc_rot = np.zeros(fcc.shape, dtype=np.complex128)
    cnt = 0
    for rr in range(ngridr):
        tmp = fcc[:, cnt:cnt + ngridc[rr]]
        ffcc = fft(tmp)

        n_theta = ngridc[rr]
        wth = ifftshift(np.arange(-n_theta/2, n_theta/2))
        mul = np.exp(-1j * wth * rgamma[:, np.newaxis])
        ffcc_rot = ffcc * mul
        tmp = ifft(ffcc_rot)
        fcc_rot[:, cnt:cnt + ngridc[rr]] = tmp

        cnt += ngridc[rr]

    return fcc_rot


def pft_phase_shift(sh, pf_grid):
    all_psi = pf_grid['all_psi']
    quad_xnodesr = pf_grid['all_r']

    phase = (np.cos(all_psi) * sh[:, np.newaxis, 0]
             + np.sin(all_psi) * sh[:, np.newaxis, 1])
    cmul = np.exp(-1j * quad_xnodesr * phase)

    return cmul


def translate_pft(fcc, sh, pf_grid):
    cmul = pft_phase_shift(sh, pf_grid)

    return fcc * cmul


def pft_norm(Mhat, pf_grid):
    quad_wts = pf_grid['quad_wts']

    return np.sqrt(np.sum((np.abs(Mhat) ** 2) * quad_wts, axis=-1))


def pft_to_fb(Shat, pf_grid):
    ngridr = pf_grid['ngridr']
    n_psi = pf_grid['n_psi']
    quad_wts = pf_grid['quad_wts']

    n_templates = Shat.shape[0]

    quad_wts_sq = quad_wts.reshape((ngridr, n_psi))

    Shathat = Shat.reshape((n_templates, ngridr, n_psi))
    # Shathat: (te, k, γ)
    Shathat = np.fft.fftshift(np.fft.fft(Shathat, axis=-1), axes=-1)
    Shathat = Shathat * quad_wts_sq[np.newaxis, :, :]
    # Shathat: (te, k, q)

    # There was a 2π factor missing before. Let's remove it.
    Shathat = Shathat / (2 * np.pi)

    return Shathat


def make_tensor_grid(rmax, ngridr, n_psi):
    dr = rmax/ngridr
    xnodesr = dr*np.arange(1, ngridr+1)
    weights = dr*np.ones(ngridr)

    psi = 2 * np.pi / n_psi * np.arange(n_psi)
    all_psi = np.repeat(psi[np.newaxis, :], ngridr, axis=0)
    all_psi = np.ravel(all_psi)

    all_r = np.repeat(xnodesr[:, np.newaxis], n_psi, axis=1)
    all_r = np.ravel(all_r)

    wts_theta = 2 * np.pi / n_psi
    quad_wts = wts_theta * xnodesr * weights
    quad_wts = np.repeat(quad_wts[:, np.newaxis], n_psi, axis=-1)
    quad_wts = np.ravel(quad_wts)

    wx = np.zeros(n_psi * ngridr)
    wy = np.zeros(n_psi * ngridr)

    cnt = 0
    for rr in range(ngridr):
        dd = xnodesr[rr]
        theta = 2 * np.pi / n_psi * np.arange(n_psi)
        wx[cnt:cnt + n_psi] = dd * np.cos(theta)
        wy[cnt:cnt + n_psi] = dd * np.sin(theta)
        cnt = cnt + n_psi

    grid = dict()
    grid['rmax'] = rmax
    grid['ngridr'] = ngridr
    grid['n_psi'] = n_psi
    grid['xnodesr'] = xnodesr
    grid['all_psi'] = all_psi
    grid['all_r'] = all_r
    grid['quad_wts'] = quad_wts
    grid['wx'] = wx
    grid['wy'] = wy

    return grid


def make_adaptive_grid(delta_range, dx, oversampling):
    all_delta = dx / oversampling * np.arange(oversampling * delta_range + 1e-10)
    n_delta = all_delta.shape[0]

    n_omega = oversampling * np.int32(np.ceil(2 * np.pi / dx * all_delta))

    n_trans = np.sum(n_omega)

    trans = np.zeros((n_trans, 2))

    cnt = 0
    for kk in range(n_delta):
        n_om = n_omega[kk]
        all_om = 2 * np.pi * np.arange(n_om) / n_om
        trans[cnt:cnt + n_om, 0] = all_delta[kk] * np.cos(all_om)
        trans[cnt:cnt + n_om, 1] = all_delta[kk] * np.sin(all_om)
        cnt += n_om

    grid = dict()
    grid['all_delta'] = all_delta
    grid['n_delta'] = n_delta
    grid['n_omega'] = n_omega
    grid['n_trans'] = n_trans
    grid['trans'] = trans

    return grid


def make_cartesian_grid(delta_range, dx, oversampling):
    Nkeep = 2 * oversampling * delta_range

    xfine = dx * np.arange(-Nkeep // 2, Nkeep // 2)

    trans = xfine
    trans = np.meshgrid(trans, trans, indexing='ij')
    trans = np.stack(trans[::-1], -1)
    trans = trans.reshape((Nkeep ** 2, 2))

    grid = {'n_trans': Nkeep ** 2, 'trans': trans}

    return grid


def extract_alignments(inner_prods3, tr_grid):
    n_images = inner_prods3.shape[0]
    n_templates = inner_prods3.shape[1]
    n_psi = inner_prods3.shape[2]
    n_trans = inner_prods3.shape[3]

    trans = tr_grid['trans']

    inner_prods3 = inner_prods3.reshape((n_images,
                                         n_templates*n_psi*n_trans))

    est_template_ind = np.zeros(n_images, dtype=np.int32)
    est_trans = np.zeros((n_images, 2))
    est_gamma = np.zeros(n_images)

    idx = inner_prods3.argmax(axis=-1)

    for cn in range(n_images):
        I3, I2, I1 = np.unravel_index(idx[cn],
                                      (n_templates, n_psi, n_trans))
        shiftx = trans[I1, 0]
        shifty = trans[I1, 1]
        rgamma = I2 * 2 * np.pi / n_psi

        est_template_ind[cn] = I3
        est_trans[cn, 0] = shiftx
        est_trans[cn, 1] = shifty
        est_gamma[cn] = rgamma

    return est_template_ind, est_trans, est_gamma


def rotations_brute_force(fimages, Shat, n_gamma, pf_grid, Nfine):
    eval_results = False

    if Shat.ndim == 2:
        Shat = Shat[np.newaxis, :, :]

    n_images, N, _ = fimages.shape
    n_templates, ngridr, ngridp = Shat.shape

    quad_wts_sq = pf_grid['quad_wts'].reshape((ngridr, ngridp))

    wx = pf_grid['wx']
    wy = pf_grid['wy']

    all_gamma = 2 * np.pi / n_gamma * np.arange(n_gamma)

    tmr_start = time()

    Shathat = fft(Shat) / ngridp
    # Shat: (te, k, γ)
    # Shathat: (te, k, q)

    Shathat = Shathat.reshape((n_templates, 1, ngridr, ngridp))
    # Shathat: (te, 1, k, q)

    wth = ifftshift(np.arange(-ngridp / 2, ngridp / 2))
    mul = np.exp(-1j * wth[np.newaxis, :] * all_gamma[:,np.newaxis])
    # mul: (γ, q)

    Shathat_rot = Shathat * mul[:, np.newaxis, :]
    # Shathat_rot: (te, γ, k, q)

    # NOTE: This can be sped up by using PyFFTW. However, for the execution to
    # be efficent, the plan must be created using FFTW_MEASURE, which takes a
    # long time. The solution will be to separate this our to the BFR
    # “planning” stage for some fixed number of images–template pairs, then
    # loop over these, computing the IFFT batchwise at execution (since the
    # exact number of pairs is not known as planning time).
    Shat_rot = ifft(Shathat_rot)

    fx1 = quad_wts_sq * Shat_rot

    fx1 = fx1.reshape((n_gamma * n_templates, ngridr*ngridp))  # for 2d1
    
    T = 2
    dx = dy = T / N

    templates_rot = np.empty((n_gamma * n_templates, N, N),
                             dtype=np.complex128)

    upsampfac = 1.25
    isign = 1
    eps = 1e-2

    finufft.nufft2d1(wx * dx, wy * dy, fx1, (N,N), isign=isign, eps=eps,
                           out=templates_rot, upsampfac=upsampfac)

    #print(templates_rot.shape)
    templates_rot = templates_rot.reshape(n_templates, n_gamma,N,N) / (4 * np.pi ** 2)
    # templates_rot: (trx, try, γ, te)  ###? no
    
    #templates_rot = templates_rot.transpose((3, 2, 1, 0)).copy()
    templates_rot = templates_rot.transpose((0,1,3,2)).copy()
    # templates_rot: (te, γ, try, trx)  
    
    ftemplates_rot = fft2(ifftshift(templates_rot, axes=(-2, -1)))
    # ftemplates_rot: (te, γ, trky, trkx)

    precomp = time() - tmr_start

    tmr_start = time()

    ftemplates_rot = ftemplates_rot[:, np.newaxis, :, :, :]
    # ftemplates_rot: (te, im, γ, trky, trkx)

    fxx = fimages[:, np.newaxis, :, :] * np.conj(ftemplates_rot)
    # ftemplates_rot: (te, im, γ, trky, trkx)

    inner_prods = pyfftw.zeros_aligned((n_templates, n_images, n_gamma, Nfine, Nfine), dtype='complex128')

    inner_prods[:, :, :, :N // 2, :N // 2] = fxx[:, :, :, :N // 2, :N // 2]
    inner_prods[:, :, :, :N // 2, -N // 2:] = fxx[:, :, :, :N // 2, -N // 2:]
    inner_prods[:, :, :, -N // 2:, :N // 2] = fxx[:, :, :, -N // 2:, :N // 2]
    inner_prods[:, :, :, -N // 2:, -N // 2:] = fxx[:, :, :, -N // 2:, -N // 2:]

    plan = pyfftw.FFTW(inner_prods, inner_prods, axes=(-2, -1),
                       direction='FFTW_BACKWARD',
                       flags=('FFTW_MEASURE',), threads=12)

    plan()

    inner_prods = np.real(inner_prods)
    inner_prods *= (Nfine / N) ** 2
    # inner_prods: (te, im, γ, try, trx)

    comp = time() - tmr_start

    return inner_prods, precomp, comp


def calc_ftk_svd(n_bessel, eps, pf_grid, tr_grid):
    all_UU = [None] * (2 * n_bessel + 1)
    all_SSVV = [None] * (2 * n_bessel + 1)
    all_rnks = np.zeros(2 * n_bessel + 1, dtype=np.int32)

    xnodesr = pf_grid['xnodesr']
    all_delta = tr_grid['all_delta']
    n_delta = tr_grid['n_delta']
    n_omega = tr_grid['n_omega']
    n_trans = tr_grid['n_trans']

    for qp in range(-n_bessel, n_bessel + 1):
        J_n = besselj(qp, -all_delta[:, np.newaxis] * xnodesr[np.newaxis, :])

        U, S, Vh = np.linalg.svd(J_n)

        ind = S > eps
        rnk = sum(ind)
        all_rnks[qp + n_bessel] = rnk

        all_UU[qp + n_bessel] = U[:, :rnk]
        all_SSVV[qp + n_bessel] = S[:rnk, np.newaxis] * Vh[:rnk, :]

    SSVV_big = np.concatenate(all_SSVV, axis=0)

    UUU = np.concatenate(all_UU, axis=1)

    all_omega = np.concatenate([2 * np.pi / n_om * np.arange(n_om)
                                for n_om in n_omega if n_om > 0])

    all_qp = np.concatenate([(k - n_bessel) * np.ones(n)
                             for k, n in enumerate(all_rnks)])

    vec_omega = np.exp(1j * all_qp[np.newaxis, :]
                       * (all_omega[:, np.newaxis] - np.pi / 2))

    BigMul_left = np.zeros((sum(all_rnks), n_trans), dtype=np.complex128)
    cnt = 0
    for kk in range(n_delta):
        n_om = n_omega[kk]
        BigMul_left[:, cnt:cnt + n_om] = (UUU[kk, :][np.newaxis, :].T
                                         * vec_omega[cnt:cnt + n_om, :].T)
        cnt += n_om

    return all_rnks, BigMul_left, SSVV_big


def premult_right_fb(Shathat, SSVV_big, all_rnks):
    n_psi = Shathat.shape[2]
    ngridr = Shathat.shape[1]
    n_templates = Shathat.shape[0]

    Shathat = Shathat.transpose((2, 0, 1))
    Shathat = Shathat.reshape((1, n_psi * n_templates, ngridr))

    SSS = SSVV_big[:, np.newaxis, :] * Shathat.conj()

    SSS = SSS.reshape((sum(all_rnks), n_psi, n_templates, ngridr))

    return SSS


def bft_plan(tr_grid, pf_grid):
    plan = {'tr_grid': tr_grid,
            'pf_grid': pf_grid}

    return plan


def bft_execute(plan, Mhat, Shat):
    pf_grid = plan['pf_grid']
    tr_grid = plan['tr_grid']

    n_psi = pf_grid['n_psi']

    Mnorm = pft_norm(Mhat, pf_grid)
    Snorm = pft_norm(Shat, pf_grid)
    MSnorm = Mnorm[:, np.newaxis] * Snorm[np.newaxis, :]

    tmr_start = time()
    Shathat = pft_to_fb(Shat, pf_grid)
    precomp1 = time() - tmr_start

    zprods1, ptm, tm = translations_brute_force_batch(Shathat, Mhat,
                                                      pf_grid, tr_grid, n_psi)

    precomp1 += ptm

    inner_prods3 = zprods1 / MSnorm[..., np.newaxis, np.newaxis]

    return inner_prods3, (precomp1, tm)


def ftk_plan(tr_grid, pf_grid, n_bessel, eps):
    all_rnks, BigMul_left, SSVV_big = calc_ftk_svd(n_bessel, eps, pf_grid, tr_grid)

    plan = {'tr_grid': tr_grid,
            'pf_grid': pf_grid,
            'n_bessel': n_bessel,
            'eps': eps,
            'all_rnks': all_rnks,
            'BigMul_left': BigMul_left,
            'SSVV_big': SSVV_big}

    return plan


def ftk_execute(plan, Mhat, Shat):
    pf_grid = plan['pf_grid']
    SSVV_big = plan['SSVV_big']
    all_rnks = plan['all_rnks']
    n_bessel = plan['n_bessel']
    BigMul_left = plan['BigMul_left']

    Mnorm = pft_norm(Mhat, pf_grid)
    Snorm = pft_norm(Shat, pf_grid)
    MSnorm = Mnorm[:, np.newaxis] * Snorm[np.newaxis, :]

    tmr_start = time()
    Shathat = pft_to_fb(Shat, pf_grid)
    SSS = premult_right_fb(Shathat, SSVV_big, all_rnks)
    precomp2 = time() - tmr_start

    zprods4, ptm, tm = svd_decomposition_alignment(SSS, Mhat, n_bessel,
                                                   all_rnks, BigMul_left)
    precomp2 += ptm

    inner_prods4 = np.real(zprods4) / MSnorm[..., np.newaxis, np.newaxis]

    return inner_prods4, (precomp2, tm)


def bfr_plan(Nfine, Nkeep, n_gamma, pf_grid, T, N):
    plan = {'Nfine': Nfine,
            'Nkeep': Nkeep,
            'n_gamma': n_gamma,
            'pf_grid': pf_grid,
            'T': T,
            'N': N}

    # TODO: FFTW plans, etc.

    return plan


def bfr_execute(plan, Mhat, Shat):
    pf_grid = plan['pf_grid']
    T = plan['T']
    N = plan['N']
    Nfine = plan['Nfine']
    Nkeep = plan['Nkeep']
    n_gamma = plan['n_gamma']

    ngridr = pf_grid['ngridr']
    n_psi = pf_grid['n_psi']

    n_templates = Shat.shape[0]
    n_images = Mhat.shape[0]

    dx = dy = T / N

    images = pft_to_cartesian(Mhat, T, N, pf_grid) / (dx * dy)

    Mnorm = pft_norm(Mhat, pf_grid)
    Snorm = pft_norm(Shat, pf_grid)

    fimages = fft2(ifftshift(images, axes=(-2, -1)))

    SShat = Shat.reshape((n_templates, ngridr, n_psi))

    fimages = fimages / Mnorm[:, np.newaxis, np.newaxis]
    SShat = SShat / Snorm[:, np.newaxis, np.newaxis]

    precomp3 = 0
    comp3 = 0

    inner_prods = np.zeros((n_images, n_templates, n_gamma, Nkeep, Nkeep), dtype=np.complex128)

    for tt in range(n_templates):
        inn, precomp, comp = rotations_brute_force(fimages, SShat[tt],
                                                   n_gamma, pf_grid, Nfine)

        # NOTE: The following truncates *and* inverts the FFT shift.
        inner_prods[:, tt, :, -Nkeep // 2:, -Nkeep // 2:] = inn[:, :, :, :Nkeep // 2, :Nkeep // 2]
        inner_prods[:, tt, :, -Nkeep // 2:, :Nkeep // 2] = inn[:, :, :, :Nkeep // 2, -Nkeep // 2:]
        inner_prods[:, tt, :, :Nkeep // 2, -Nkeep // 2:] = inn[:, :, :, -Nkeep // 2:, :Nkeep // 2]
        inner_prods[:, tt, :, :Nkeep // 2, :Nkeep // 2] = inn[:, :, :, -Nkeep // 2:, -Nkeep // 2:]

        precomp3 += precomp
        comp3 += comp

    inner_prods = inner_prods.reshape((n_images, n_templates, n_gamma, Nkeep ** 2))

    return inner_prods, (precomp3, comp3)
