# Quick-start Guide to QuickCW

## Setup
`QuickCW` relies on the following non-standard packages:
```
enterprise
enterprise_extensions
numba
numba_stats
```

Using `conda`, we can set up a new clean environment with these as follows. Create new `conda` environment:
```
conda create --name QuickCW python=3.9
```
Active our new environment:
```
conda activate QuickCW
```
Install `enterprise`:
```
conda install -c conda-forge enterprise-pulsar
```
Install `enterprise_extensions`:
```
pip install git+https://github.com/nanograv/enterprise_extensions@master
```
Install `numba`:
```
conda install numba
```
Install v1.0.1 of `numba_stats` (make sure to use this because the latest release v1.1.0 breaks things):
```
pip install numba-stats==1.0.1
```

Once these are installed `QuickCW` can be used without installation.

## Running QuickCW
The main analysis code can be found in `QuickCW.py`, which can be executed by the wrapper script `runQuickMCMC.py`. Before running, make sure that in that script:
* `data_pkl` points to the pickled pulsar object you want to analyze. Alternatively one can rewrite the script so that it loads in par/tim files. What matters in the end is that `psrs` contain the pulsar objects we want to use.
* `noisefile` points to the json file containing the noise dictionary we plan to use for setting the white noise parameters.
* `savefile` is the name of the file we want to save our results.

Once these are set, we can run the MCMC by executing:

```
python runQuickMCMC.py
```
If we want to run this in the background, we can execute:
```
nohup ./run_QuickCW.sh > nohup.out 2>&1 &
```

This will use `run_QuickCW.sh` to run the analysis in the background and send all output into the file `nohup.out`.

## Postprocessing
Once the MCMC run finished (or even during since it saves intermediate results during runtime) all the results can be found in a single HDF5 file. Follow this jupyter notebook for a few simple postprocessing of the results, like traces, corner plots and upper limit curves: https://github.com/bencebecsy/QuickCW/blob/main/doc/plotting_results.ipynb

