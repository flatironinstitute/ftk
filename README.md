# Factorization of the translation kernel for fast rigid image alignment

This repository contains code to reproduce the benchmarks in [the paper](https://arxiv.org/abs/1905.12317). To run the benchmarks and generate the plots, install the dependencies using
```
pip3 install -r requirements.txt
```
then run
```
run.sh
```
Upon completion, the benchmarks are stored in the `results` directory while EPS figures are in the `figures` directory.

Note: the benchmark takes several (>10) minutes, and requires a workstation with at least 128 GB RAM.
