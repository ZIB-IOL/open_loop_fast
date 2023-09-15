# Affine-Invariant Convergence Rates of the Frank-Wolfe Algorithm with Open-Loop Step-Sizes

## References

This project is an extension of the previously published Git repository
[open_loop_fw](https://github.com/ZIB-IOL/open_loop_fw), which contains the code to the following paper:

Wirth, E., Pokutta, S., and Kerdreux, T. (2023). Acceleration of Frank-Wolfe Algorithms with Open-Loop Step-Sizes. 
In Proceedings of AISTATS.


## Installation guide

Download the repository and store it in your preferred location, say ~/tmp.

Open your terminal and navigate to ~/tmp.

Run the command:
```shell script
$ conda env create --file environment.yml
```

This will create the conda environment affine_invariant_open_loop_fw.

Activate the conda environment with:
```shell script
$ conda activate affine_invariant_open_loop_fw
```
Navigate to ~/tmp

To perform the experiments in the paper:

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
>>> python3 -m experiments.regression
```
```python3 script
>>> python3 -m experiments.strong_growth
```
```python3 script
>>> python3 -m experiments.weak_boundary_growth
```



The experiments are then stored in ~/tmp/experiments/figures.