from hexapod.controllers.NEATController import Controller, tripod_gait, reshape, stationary
from hexapod.simulator import Simulator
import pymap_elites.map_elites.cvt as cvt_map_elites
import numpy as np
import neat
import pymap_elites.map_elites.common as cm
import pickle
import os

"""
A script used to test the NEAT Map Elites experiments. 
"""

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

# Name of map to load in
filename = 'mapElitesOutput/NEAT/0_20000archive/archive_genome8011476.pkl'
with open(filename, 'rb') as f:
    genomes = pickle.load(f)

# Specify failure scenario
failed_legs = []

# The genome in the map that we want to test
test = genomes[225]
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
