import os
from scipy.stats import norm
import numpy as np
import gzip

rand = np.random.random


def randn(sz):
    return norm.ppf(rand(sz))


def sym_rand(sz):
    sgn = np.sign(randn(sz))
    return sgn * rand(sz)


def load_images(n_images):
    K = 10
    N = 128

    data_dir = get_data_dir()
    filename = os.path.join(data_dir, 'images.bin.gz')

    with open(filename, 'rb') as f:
        buf = f.read()
    buf = gzip.decompress(buf)

    m = np.frombuffer(buf, dtype=np.float32)
    m = m.reshape((K, N, N))

    return m[:n_images, :, :]


def gen_rot_trans(n_images, delta_max):
    true_gamma = np.zeros(n_images)
    true_trans = np.zeros((n_images, 2))

    for k in range(n_images):
        true_gamma[k] = 2 * np.pi * rand()
        true_trans[k, 0] = sym_rand(1) * delta_max
        true_trans[k, 1] = sym_rand(1) * delta_max

    return true_gamma, true_trans


def get_root_dir():
    src_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.dirname(src_dir)

    return root_dir


def get_data_dir():
    root_dir = get_root_dir()
    data_dir = os.path.join(root_dir, 'data')

    return data_dir


def get_results_dir(create_dir=True):
    root_dir = get_root_dir()
    results_dir = os.path.join(root_dir, 'results')

    if create_dir and not os.path.exists(results_dir):
        os.mkdir(results_dir)

    return results_dir


def get_figures_dir(create_dir=True):
    root_dir = get_root_dir()
    figures_dir = os.path.join(root_dir, 'figures')

    if create_dir and not os.path.exists(figures_dir):
        os.mkdir(figures_dir)

    return figures_dir
