## Beamtime  
This repo is predominantly for analysis of the lcls x42619 beamtime performed end of June 2021.  
This will also host code that analyzes for the xcomm118 for runs in the 400's since that was the commissioning shifts.  

# setting up  
First set the configuration .h5 file with something like this...
```bash
./src/set_configs.py /reg/data/ana16/tmo/tmox42619/scratch/ryan_output_2022/h5files/tmox42619.hsdconfig.h5
```
Here one must be sure that the file written is indeed what will be read in ```./src/hits2h5.py```
the ```expname``` gets used to pull the config file, at least until we implement the parser.  

```bash
scratchpath=/reg/data/ana16/tmo/tmox42619/scratch/ryan_output_2022/h5files
expname=tmox42619
runnum=23
shots=100
./src/set_configs.py ${scratchpath}/${expname}.hsdconfig.h5
./src/hits2h5.py ${expname} ${runnum} ${nshots}
```

# using slurm  
```bash
sbatch -p psanaq --nodes 1 --ntasks-per-node 1 --wrap="./src/hits2h5.py tmox42619 23 10000"
```

# Fitting Argon Calibrations
The first line creates the calibration file based on the hard coded *hand calibrated-by ryan* values for the argon Auger-Meitner lines near 200-210eV (for 600eV photon energy so far).  
The second line then fits those values and writes the resulting quadratic fit to a file 'calibfit.h5'.  
```bash
./src/argon.calib.auger.py /media/coffee/dataSD/ryan_output_2022/calib/argon.calib.20220411.h5
./src/argon.calib.auger.fitting.py /media/coffee/dataSD/ryan_output_2022/calib/argon.calib.20220411.h5
```
Inside ```./src/argon.calib.auger.fitting.py``` there are both fit() and transform() methods.  These I plan to make compatible with import for post processing the 'tofs' arrays in the hits2h5.py result.
