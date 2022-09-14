from hexapod.controllers.NEATController import Controller, reshape, stationary
from hexapod.simulator import Simulator
import pymap_elites.map_elites.cvt as cvt_map_elites
import numpy as np
import neat
import pymap_elites.map_elites.common as cm
import pickle
import os
import sys

"""
A script used to test the NEAT Map Elites experiments. 
"""

# Maps to be tested
maps20Genome = ["mapElitesOutput/NEAT/0_20000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/1_20000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/2_20000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/3_20000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/4_20000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/5_20000archive/archive_genome8001916.pkl",
                "mapElitesOutput/NEAT/6_20000archive/archive_genome8001916.pkl",
                "mapElitesOutput/NEAT/7_20000archive/archive_genome8001916.pkl",
                "mapElitesOutput/NEAT/8_20000archive/archive_genome8001916.pkl",
                "mapElitesOutput/NEAT/9_20000archive/archive_genome8001916.pkl"
                ]


maps40Genome = ["mapElitesOutput/NEAT/0_40000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/1_40000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/2_40000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/3_40000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/4_40000archive/archive_genome8011476.pkl",
                "mapElitesOutput/NEAT/5_40000archive/archive_genome8001916.pkl",
                "mapElitesOutput/NEAT/6_40000archive/archive_genome8001916.pkl",
                "mapElitesOutput/NEAT/7_40000archive/archive_genome8001916.pkl",
                "mapElitesOutput/NEAT/8_40000archive/archive_genome8001916.pkl",
                "mapElitesOutput/NEAT/9_40000archive/archive_genome8001916.pkl"
                ]


# Load config file for ANN
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'NEATHex/config-feedforward')

# Fitness function
def evaluate_gait(x, failed_legs, duration=5):
    # Set up neural network
    net = neat.nn.FeedForwardNetwork.create(x, config)

    # Load initial stationary legs
    leg_params = np.array(stationary).reshape(6, 5)
    # Set up controller
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
    return fitness, descriptor


# Choose map type, map number, genome number, failure scenario
mapType = int(sys.argv[1]) # Map type
mapNumber = int(sys.argv[2]) # Map Number 
genomeNumber = int(sys.argv[3]) # Genome number
try:
    failed1 = int(sys.argv[4]) # Failed leg 1
except:
    failed_legs=[]
try:
    failed2 = int(sys.argv[5]) # Failed leg 2
    failed_legs = [failed1, failed2]
except:
    failed_legs=[failed1]
if mapType == 0:
    filename = maps20Genome[mapNumber]
else:
    filename = maps40Genome[mapNumber]

# Map to load in
with open(filename, 'rb') as f:
    genomes = pickle.load(f)

# The genome in the map that we want to test
test = genomes[genomeNumber]
# Print fitness and behavioural descriptor of genome
print(evaluate_gait(test, duration = 5, failed_legs = failed_legs))

# Create neural network from genome
winner_net = neat.nn.FeedForwardNetwork.create(test, config)

# Create and run controller
controller = Controller(stationary, body_height=0.15, velocity=0.5, crab_angle=-np.pi / 6, ann=winner_net,
                            printangles=True)
simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=failed_legs)


while True:
    simulator.step()
