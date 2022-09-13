import math
import pickle
import neat
import visualize
import neat.nn
import numpy as np
import multiprocessing
import os
import sys
import visualize as vz
import shutil
from hexapod.controllers.hyperNEATController import Controller, reshape, stationary
from hexapod.simulator import Simulator
from pureples.hyperneat import create_phenotype_network
from pureples.shared import Substrate, run_hyper
from pureples.shared.visualize import draw_net
import pymap_elites.map_elites.common as cm
import pymap_elites.map_elites.cvt as cvt_map_elites

"""
A script used to test the HyperNEAT Map Elites experiments. 
"""

# Fitness function
def evaluate_gait(x, failed_legs, duration=5):
    cppn = neat.nn.FeedForwardNetwork.create(x, CONFIG)
    # Create ANN from CPPN and Substrate
    net = create_phenotype_network(cppn, SUBSTRATE)
    leg_params = np.array(stationary).reshape(6, 5)
    # Setup controller
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
                                ann=net)
    except:
        return 0, np.zeros(6)
    # Initialise Simulator
    simulator = Simulator(controller=controller, visualiser=False, collision_fatal=False, failed_legs=failed_legs)
    # Step in simulator
    contact_sequence = np.full((6, 0), False)
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            fitness = 0, np.zeros(6)
        contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0,
                               posinf=0.0, neginf=0.0)
    # Terminate Simulator
    simulator.terminate()
    # Assign fitness to genome
    x.fitness = fitness
    return fitness, descriptor

# Define HyperNEAT Substrate
INPUT_COORDINATES = [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5),
                     (0.2, 0), (0.4, 0), (0.6, 0),
                     (0.2, -0.5), (0.4, -0.5), (0.6, -0.5),
                     (-0.6, -0.5), (-0.4, -0.5), (-0.2, -0.5),
                     (-0.6, 0), (-0.4, 0), (-0.2, 0),
                     (-0.6, 0.5), (-0.4, 0.5), (-0.2, 0.5),
                     (0, 0.25), (0, -0.25)]
OUTPUT_COORDINATES = [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5),
                     (0.2, 0), (0.4, 0), (0.6, 0),
                     (0.2, -0.5), (0.4, -0.5), (0.6, -0.5),
                     (-0.6, -0.5), (-0.4, -0.5), (-0.2, -0.5),
                     (-0.6, 0), (-0.4, 0), (-0.2, 0),
                     (-0.6, 0.5), (-0.4, 0.5), (-0.2, 0.5)]
HIDDEN_COORDINATES = [[(0.2, 0.5), (0.4, 0.5), (0.6, 0.5),
                     (0.2, 0), (0.4, 0), (0.6, 0),
                     (0.2, -0.5), (0.4, -0.5), (0.6, -0.5),
                     (-0.6, -0.5), (-0.4, -0.5), (-0.2, -0.5),
                     (-0.6, 0), (-0.4, 0), (-0.2, 0),
                     (-0.6, 0.5), (-0.4, 0.5), (-0.2, 0.5)]]

# Pass configuration to substrate
SUBSTRATE = Substrate(
    INPUT_COORDINATES, OUTPUT_COORDINATES, HIDDEN_COORDINATES)
ACTIVATIONS = len(HIDDEN_COORDINATES) + 2

# Configure cppn using config file
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'NEATHex/config-cppn')

# Filename of map to load in
filename = "mapElitesOutput/HyperNEAT/0_20000archive/archive_genome8011438.pkl"
with open(filename, 'rb') as f:
    genomes = pickle.load(f)

# Specify failure scenario
failed_legs = [1, 2]
# The genome we want to test
test = genomes[224]
# Print fitness of genome
print(evaluate_gait(test, failed_legs = failed_legs))
# Setup CPPN
CPPN = neat.nn.FeedForwardNetwork.create(test, CONFIG)
# Optional Drawing of CPPN
# vz.draw_net(CONFIG, test)

# Setup ANN from CPPN and Substrate
WINNER_NET = create_phenotype_network(CPPN, SUBSTRATE)

# Create and run controller
controller = Controller(stationary, body_height=0.15, velocity=0.5, crab_angle=-np.pi / 6, ann=WINNER_NET, activations=ACTIVATIONS)
simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=failed_legs)

while True:
    simulator.step()