# Quick-start Guide to QuickCW

## Setup
It is advisable, to create new `conda` environment:
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
Clone the `QuickCW` repo:
```
git clone https://github.com/bencebecsy/QuickCW.git
```
Move into the repo's folder:
```
cd QuickCW
```
Install `QuickCW` and all remaining requirements:
```
pip install -e .
```

## Running QuickCW
The main analysis code can be found in `QuickCW.py`, which can be executed by the wrapper script `runQuickMCMC.py`. Move this file into the folder where you want to run it, and modify it so that:
* `data_pkl` points to the pickled pulsar object you want to analyze. Alternatively one can rewrite the script so that it loads in par/tim files. What matters in the end is that `psrs` contain the pulsar objects we want to use.
* `noisefile` points to the json file containing the noise dictionary we plan to use for setting the white noise parameters.
* `savefile` is the name of the file we want to save our results.
* The number of iterations (`N`) is set properly. The example script has `N=1_000_000`, which is good for a quick test run to see that everything works, but not enough for an actual analysis. Depending on the details of the analysis and the dataset one might want to set at least `N=100_000_000` (or even `N=1_000_000_000`). Note that the number of steps in the shape parameters is `N/n_int_block`, so for example the example script gives '1_000_000/10_000=100' steps in shape parameters.
* If using a higher `N` than in the example script, it can also be useful to set `thin=100` (or even 1000), which results in only saving every 100th/1000th sample to file and thus helps keep file sizes down (default is 10).

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
Once the MCMC run finished (or even during since it saves intermediate results during runtime) all the results can be found in a single HDF5 file. Follow this jupyter notebook for a few simple postprocessing of the results, like traces, corner plots and upper limit curves: https://github.com/bencebecsy/QuickCW/blob/main/docs/plotting_results.ipynb

