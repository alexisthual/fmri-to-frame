# Functional alignment of MRI signals decodes visual semantics across species

This repo contains a series of python files to produce results presented in the paper [Functional alignment of MRI signals decodes visual semantics across species]().
All computations were performed on a Slurm cluster.

This work shows that it is possible to train models that decode semantic visual perceptions in human participants, and partially transfer them to non-human primates.

![Introduction figure](https://raw.githubusercontent.com/alexisthual/fmri-to-frame/main/figures/intro.png)

## Installation

## File structure

Files are organised in two categories:

1. `./src/scripts` contains standalone jobs and utility scripts. Some tasks performed include:
    * computing a cortical alignment between two participants
    * computing latent representations of stimuli seen by participants
    * training a brain decoder
    * evaluating a brain decoder

2. `./scr/launchers` contains scripts which launch a series of standalone jobs in parallel, typically on a Slurm cluster.

## Acknowledgement

This work was performed using HPC resources from GENCI-IDRIS (Jean-Zay cluster, project AD010613496R1).

A.T, S. D and B.T's research has received funding from the European Union's Horizon 2020 Framework Programme for Research and Innovation under the Specific Grant Agreement No. 945539 (Human Brain Project SGA3). It has also been supported by the the KARAIB AI chair (ANR-20-CHIA-0025-01) and the NeuroMind Inria associate team.