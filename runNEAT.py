import neat
from hexapod.controllers.NEATController import Controller, reshape, stationary
from hexapod.simulator import Simulator
import numpy as np
import multiprocessing
import os
import sys
import shutil
import pickle
import visualize as vz
from pureples.shared.visualize import draw_net

"""
A script used Run a single NEAT experiment.

Takes two command line arguments.
1) The runNumber
2) The number of Generations to run the algorithm for
"""

# Fitness Function
def evaluate_gait(genomes, config, duration=5):
    for genome_id, genome in genomes:
        # Set up neural net
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        leg_params = np.array(stationary).reshape(6, 5)

        # Setup Controller and Simulator
        try:
            controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
                                    ann=net)
        except:
            return 0, np.zeros(6)
        simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
        # Step in Simulator
        for t in np.arange(0, duration, step=simulator.dt):
            try:
                simulator.step()
            except RuntimeError as collision:
                fitness = 0, np.zeros(6)
        fitness = simulator.base_pos()[0]  # distance travelled along x axis
        simulator.terminate()
        genome.fitness = fitness

# Fitness Function
def evaluate_gait_parallel(genome, config, duration=5):
    # Setup Neural Network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    leg_params = np.array(stationary).reshape(6, 5)

    # Setup Controller
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6, ann=net)
    except:
        return 0, np.zeros(6)
    simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
    # Step in Simulator
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            fitness = 0, np.zeros(6)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    simulator.terminate()
    return fitness

# Load in Config File
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'NEATHex/config-feedforward')


def runNeat(gens):
    """
    Create the population and run the experiment.
    Returns the winning genome and the statistics of the run.
    """
    p = neat.Population(config)
    stats = neat.statistics.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))
    pe = neat.parallel.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_gait_parallel)
    winner = p.run(pe.evaluate, gens)

    return winner, stats


if __name__ == '__main__':
    # Setup directories for output
    if not os.path.exists("NEATOutput"):
        os.mkdir("NEATOutput")
        if not os.path.exists("NEATOutput/genomeFitness"):
            os.mkdir("NEATOutput/genomeFitness")
        if not os.path.exists("NEATOutput/graphs"):
            os.mkdir("NEATOutput/graphs")
        if not os.path.exists("NEATOutput/bestGenomes"):
            os.mkdir("NEATOutput/bestGenomes")
        if not os.path.exists("NEATOutput/stats"):
            os.mkdir("NEATOutput/stats")
    numRuns = int(sys.argv[1])
    fileNumber = (sys.argv[2])
    winner, stats = runNeat(numRuns) # Run experiment
    print('\nBest genome:\n{!s}'.format(winner))

    # Save all stats and graphs
    stats.save_genome_fitness(delimiter=',', filename='NEATOutput/genomeFitness/NEATFitnessHistory' + fileNumber + '.csv')
    vz.plot_stats(stats, ylog=False, view=True, filename='NEATOutput/graphs/NEATAverageFitness' + fileNumber + '.svg')
    vz.plot_species(stats, view=True, filename='NEATOutput/graphs/NEATSpeciation' + fileNumber + '.svg')

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    outputNameGenome = "NEATGenome" + fileNumber + ".pkl"
    outputNamePopulation = "NEATStats" + fileNumber + ".pkl"

    with open('NEATOutput/bestGenomes/' + outputNameGenome, 'wb') as output:
        pickle.dump(winner, output, pickle.HIGHEST_PROTOCOL)
    with open('NEATOutput/stats/' + outputNamePopulation, 'wb') as output:
        pickle.dump(stats, output, pickle.HIGHEST_PROTOCOL)
    draw_net(winner_net, filename="NEATOutput/graphs/NEATWINNER" + fileNumber)

    controller = Controller(stationary, body_height=0.15, velocity=0.5, crab_angle=-1.57, ann=winner_net,
                            printangles=True)
    simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=[0])

    while True:
        simulator.step()
