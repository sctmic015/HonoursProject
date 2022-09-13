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

3) Run a genome from the NEAT maps
```
python3 MapElitesFitnessTest.py
```

The filepath to the map, genome within the map and damage conditions can be specified within the script. Performances to damage conditions were calculated using MBOA and can be found in mapElitesOutput/NEATSim.

4) Run a genome from the HyperNEAT maps
```
python3 MapElitesHyperNEATFitnessTest.py
```

The filepath to the map, genome within the map and damage conditions can be specified within the script. Performances to damage conditions were calculated using MBOA and can be found in mapElitesOutput/HyperNEATSim.

## Running Experiments

Note: All experiments take a long time to run. Upwards of 24 hours on a 24 core node at the Center of High Performance Computing. 

1) Run a NEAT Experiment
```
python3 runNEAT.py numGens fileNumber
```

Where numGens is the number of generations to run the experiment for (fixed pop size of 300) and fileNumber is used for the outputs from the experiment. 

2) Run a HyperNEAT Experiment
```
python3 runHyperNEAT.py numGens fileNumber
```

Where numGens is the number of generations to run the experiment for (fixed pop size of 300) and fileNumber is used for the outputs from the experiment. 

3) Run a MapElitesExperiment using a NEAT encoding.
```
python3 mapElitesNEAT.py mapSize runNum
```

Where mapSize is the map size used in the experiment and runNum is used for the outputs from the experiment.

4) Run a MapElitesExperiment using a HyperNEAT encoding.
```
python3 mapElitesHyperNEAT.py mapSize runNum
```

Where mapSize is the map size used in the experiment and runNum is used for the outputs from the experiment.

5) Run M-BOA for the NEAT maps
```
python3 adaptTestsNEAT.py mapType DamageScenario
```

Where mapType is a 0 or 1 indicate whether we are running the test for 20k or 40k maps.
Where DamageScenario is ann integer [0, 4] indicating which damage scenario we are testing.

6) Run M-BOA for the HyperNEAT maps
```
python3 adaptTestsHyperNEAT.py mapType DamageScenario
```

Where mapType is a 0 or 1 indicate whether we are running the test for 20k or 40k maps.
Where DamageScenario is ann integer [0, 4] indicating which damage scenario we are testing.

## Output From Experiments

1) NEATOutput

Contains all output from the NEAT experiments including genomes, graphs and fitness per generation. 

2) HyperNEATOutput

Contains all output from the HyperNEAT experiments including genomes, graphs and fitness per generation.

3) MapElitesOutut

Contains all output from the MapElites and Map Based Bayesian Optimisation Experiments.

NEAT and HyperNEAT contain Map Elites output
NEATSim and HyperNEATSim contain MBOA output
