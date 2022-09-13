# HyperNEAT for Gait Evolution and Damage Control in a Hexapod Robot

## About

This repository contains all the code and results for my Honours Project in Evolutionary Computing. The project was split into two parts:
1) The evolution of hexapod gaits using the NEAT and HyperNEAT algorithms
2) The use of Map Elites when used in conjunction with NEAT and HyperNEAT as encodings. 

## Requirements
```shell
pip install -r requirements.txt
```

* numpy
* pybullet
* matplotlib
* sklearn
* GPy
* scipy
* neat-python
* graphviz

## Videos
[Hexapod Playlist](https://youtube.com/playlist?list=PLTJmZivqOPVHI3edrhLOUiYNdeAQAcrH2)


## Running Pre Computer Experiments

1) Load best genome (gait) from the 20 NEAT experiments
```
python3 NEATLoad.py runNum
``` 

where runNum is an integer between 0 and 19 indicating which of the experiment replicates to load.

2) Load best genome (gait) from the 20 HyperNEAT experiments 
```
python3 NEATLoad.py runNum
``` 

where runNum is an integer between 0 and 19 indicating which of the experiment replicates to load.