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
from pureples.shared.visualize import draw_net

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     r'C:\Users\micha\PycharmProjects\Honours Project\NEATHex\config-feedforward')


def populateNEATANN():
    for i in range(20):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\NEATOutput\bestGenomes\NEATGenome{i}.pkl'
        with open(filename, 'rb') as f:
            winner = pickle.load(f)
            winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
            draw_net(winner_net,
                     filename=r"C:\Users\micha\PycharmProjects\Honours Project\NEATOutput\graphs\NEATWinner" + str(i))


def NEATSolutionsFromMapElites():
    filename = r'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\8_20000archive\archive_genome8001916.pkl'
    with open(filename, 'rb') as f:
        genomes = pickle.load(f)
        genome = genomes[8521]
        #winner_net = neat.nn.FeedForwardNetwork.create(config, genome)
        vz.draw_net(config, genome)

NEATSolutionsFromMapElites()
