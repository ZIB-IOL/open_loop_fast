# Fast Open-Loop Step-Size for Frank-Wolfe

## References

This project is an extension of the previously published Git repository
[ affine_invariant_open_loop_fw
Public](https://github.com/ZIB-IOL/affine_invariant_open_loop_fw), which contains the code to the following paper:

Wirth, E., Pena, J., and Pokutta, S. (2023b). Accelerated affine-invariant convergence rates of the Frank-Wolfe
algorithm with open-loop step-sizes. arXiv preprint arXiv:2310.04096.


## Installation guide

Download the repository and store it in your preferred location, say ~/tmp.

Open your terminal and navigate to ~/tmp.

Run the command:
```shell script
$ conda env create --file environment.yml
```

This will create the conda environment open_loop_fast.

Activate the conda environment with:
```shell script
$ conda activate open_loop_fast
```
Navigate to ~/tmp

To perform the experiments in the paper:

```python3 script
>>> python3 -m experiments.ablation_study_l
```
```python3 script
>>> python3 -m experiments.collaborative_filtering
```
```python3 script
>>> python3 -m experiments.gaps_growth
```
```python3 script
>>> python3 -m experiments.logistic_regression
```
```python3 script
>>> python3 -m experiments.polytope
```
```python3 script
>>> python3 -m experiments.polytope_ls_ol
```
```python3 script
>>> python3 -m experiments.regression
```
```python3 script
>>> python3 -m experiments.strong_growth
```
```python3 script
>>> python3 -m experiments.weak_boundary_growth
```

The experiments are then stored in ~/tmp/experiments/figures.