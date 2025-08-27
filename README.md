# MSSM Scanning

This repository contains most of the code used throughout my bachelor thesis. The aim is to perform MSSM scans with the open source framework
[GAMBIT](https://gambitbsm.org). The objective of the scans is to obtain models conform with the theory of Super Symmetry. The output holds information about
all the different properties of the particles which constitute the model, e.g. masses, mixing and branching fractions. The results should then be compared to
already existing scans such as the EWK scan obtained with [Run3ModelGen](https://github.com/jwuerzinger/Run3ModelGen). What makes GAMBIT interesting is its
capability to use different sampling algorithms, one of which is [Diver](https://diver.hepforge.org). Here instead of randomly selecting the input parameters for a scan from
predefined ranges like in the previously mentioned Run3ModelGen, GAMBIT will compute a likelihood after for each generated model and select possible points for the next model
based on this information. This approach creates models with ever increasing likelihood. The goal of my thesis is to compare the two approaches and categorize what might be
advantageous behaviour of the scan with Differential Evolution (Diver). To get some metrics about this I want to look at the sampling efficiency of the scans and their parameter
space coverage.

This repository is split into different parts so different concepts each get their own section where thex can be explored. In the following I will give a brief overview
about each section.

## Diver

To illustrate the concept of Differential Evolution I implemented my own version of it based on the concepts found in the Diver package from GAMBIT.
from the implementation I try to extract some useful knowledge about the parameter space coverage and I also create some animations annd plots.

To play around with it yourself you can refer to the `README` in the Diver section.

## MSSM

In this section I try to illustrate different concepts regarding the MSSM models such as the priors used for the GAMBIT scans or the minimization condition from which
the mixing of the Higgs Doublets is found in the scans. Since these concepts are mostly standalone pieces of code, I chose to implement them in notebooks so I can explain the
concepts alongside the code. 
Also here I provide two notebooks to easily inspect the ouput of the GAMBIT and the Run3ModelGen output which come in a `.hdf5` and `.root` file format respectively. These notebooks can be used to quickly access the results of a scan and gather some general information as well as wheter or not the scan was succesful.

## data

Here I will put the output of the scans I will use throughout my thesis as well as the input configurations of the corresponding scans

## Util

Everything that just makes my life easier goes here. This will be scripts to handle multiple outputfiles, transfer of data, etc

Note that alongside this repository I also created a tool to visually inspect `.hdf5` files in a web page. The source code can be found [here](https://github.com/vallereichi/hdf5-view)
And the current status of the GAMBIT configuration and its implementation into a slurm batch system can be found [here](https://github.com/vallereichi/workflow-gambit)
