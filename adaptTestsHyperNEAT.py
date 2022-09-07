from hexapod.controllers.hyperNEATController import Controller, tripod_gait, reshape, stationary
from hexapod.simulator import Simulator
from pureples.hyperneat import create_phenotype_network
from pureples.shared import Substrate, run_hyper
from adapt.MBOA import MBOA
import numpy as np
import pickle
import neat
import sys

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

# parameters
maps20File = ["mapElitesOutput/HyperNEAT/0_20000archive/archive8011438.dat",
              "mapElitesOutput/HyperNEAT/1_20000archive/archive8011438.dat",
              "mapElitesOutput/HyperNEAT/2_20000archive/archive8001878.dat",
              "mapElitesOutput/HyperNEAT/3_20000archive/archive8001878.dat",
              "mapElitesOutput/HyperNEAT/4_20000archive/archive8011438.dat",
              "mapElitesOutput/HyperNEAT/5_20000archive/archive8001878.dat",
              "mapElitesOutput/HyperNEAT/6_20000archive/archive8001878.dat",
              "mapElitesOutput/HyperNEAT/7_20000archive/archive8001878.dat",
              "mapElitesOutput/HyperNEAT/8_20000archive/archive8001878.dat",
              "mapElitesOutput/HyperNEAT/9_20000archive/archive8001878.dat"
              ]

maps20Genome = ["mapElitesOutput/HyperNEAT/0_20000archive/archive_genome8011438.pkl",
                "mapElitesOutput/HyperNEAT/1_20000archive/archive_genome8011438.pkl",
                "mapElitesOutput/HyperNEAT/2_20000archive/archive_genome8001878.pkl",
                "mapElitesOutput/HyperNEAT/3_20000archive/archive_genome8001878.pkl",
                "mapElitesOutput/HyperNEAT/4_20000archive/archive_genome8011438.pkl",
                "mapElitesOutput/HyperNEAT/5_20000archive/archive_genome8001878.pkl",
                "mapElitesOutput/HyperNEAT/6_20000archive/archive_genome8001878.pkl",
                "mapElitesOutput/HyperNEAT/7_20000archive/archive_genome8001878.pkl",
                "mapElitesOutput/HyperNEAT/8_20000archive/archive_genome8001878.pkl",
                "mapElitesOutput/HyperNEAT/9_20000archive/archive_genome8001878.pkl"
                ]

maps40File = ["mapElitesOutput/HyperNEAT/0_40000archive/archive8001878.dat",
              "mapElitesOutput/HyperNEAT/1_40000archive/archive8001878.dat",
              "mapElitesOutput/HyperNEAT/2_40000archive/archive8001878.dat",
              "mapElitesOutput/HyperNEAT/3_40000archive/archive8011438.dat",
              "mapElitesOutput/HyperNEAT/4_40000archive/archive8001878.dat"]

maps40Genome = ["mapElitesOutput/HyperNEAT/0_40000archive/archive_genome8001878.pkl",
                "mapElitesOutput/HyperNEAT/1_40000archive/archive_genome8001878.pkl",
                "mapElitesOutput/HyperNEAT/2_40000archive/archive_genome8001878.pkl",
                "mapElitesOutput/HyperNEAT/3_40000archive/archive_genome8011438.pkl",
                "mapElitesOutput/HyperNEAT/4_40000archive/archive_genome8001878.pkl"]

mapType = False  # False for 20k maps, True for 40k maps
map_count = 10
mapsFile = ''
if mapType == False:
    niches = 20000
    mapsFile = maps20File
    mapsGenome = maps20Genome
else:
    niches = 40000
    mapsFile = maps40File
    mapsGenome = maps40Genome
failure_scenario = int(sys.argv[1])

S0 = [[]]
S1 = [[1], [2], [3], [4], [5], [6]]
S2 = [[1, 4], [2, 5], [3, 6]]
S3 = [[1, 3], [2, 4], [3, 5], [4, 6], [5, 1], [6, 2]]
S4 = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]]

scenarios = [S0, S1, S2, S3, S4]
failures = scenarios[failure_scenario]

num_its = np.zeros((len(failures), map_count))
best_indexes = np.zeros((len(failures), map_count))
best_perfs = np.zeros((len(failures), map_count))

for failure_index, failed_legs in enumerate(failures):
    print("Failed legs:", failed_legs)
    for map_num in range(0, len(maps20Genome)):
        print("Testing map:", map_num)


        # need to redefine the evaluate function each time to include the failed leg
        def evaluate_gait(x, duration=5):
            cppn = neat.nn.FeedForwardNetwork.create(x, CONFIG)
            # Create ANN from CPPN and Substrate
            net = create_phenotype_network(cppn, SUBSTRATE)
            # Reset net

            leg_params = np.array(stationary).reshape(6, 5)
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
            # print(difference)
            # fitness = difference
            # Assign fitness to genome
            x.fitness = fitness
            return fitness


        mapGenome = mapsGenome[map_num]
        mapFile = mapsFile[map_num]
        with open(mapGenome, 'rb') as f:
            genomes = pickle.load(f)
        num_it, best_index, best_perf, new_map = MBOA(mapFile, genomes, f"./centroids/centroids_{niches}_6.dat",
                                                      evaluate_gait, max_iter=50, print_output=False)
        print(num_it)
        num_its[failure_index, map_num - 1] = num_it
        best_indexes[failure_index, map_num - 1] = best_index
        best_perfs[failure_index, map_num - 1] = best_perf

# np.savetxt(f"./experiments/sim/{niches}k_half/trials_{failure_scenario}.dat", num_its, '%d')
# np.savetxt(f"./experiments/sim/{niches}k_half/perfs_{failure_scenario}.dat", best_perfs)

np.savetxt(f"./mapElitesOutput/HyperNEATSim/{niches}_niches/trials_{failure_scenario}.dat", num_its, '%d')
np.savetxt(f"./mapElitesOutput/HyperNEATSim/{niches}_niches/perfs_{failure_scenario}.dat", best_perfs)
np.savetxt(f"./mapElitesOutput/HyperNEATSim/{niches}_niches/indexes_{failure_scenario}.dat", best_indexes)
