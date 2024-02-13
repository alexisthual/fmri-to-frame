# fMRI to frame

This repo contains a series of python files to reproduce results presented in [PAPER]().

These files are organised in two categories:

1. `./src/scripts` contains standalone jobs and utility scripts. Some tasks performed include:
    * computing a cortical alignment between two participants
    * computing latent representations of stimuli seen by participants
    * training a brain decoder
    * evaluating a brain decoder

2. `./scr/launchers` contains scripts which launch a series of standalone jobs in parallel, typically on a Slurm cluster.
