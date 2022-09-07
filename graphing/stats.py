from hexapod.controllers.NEATController import Controller, tripod_gait, reshape, stationary
from hexapod.controllers.hyperNEATController import Controller as HController
from hexapod.simulator import Simulator
import neat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from matplotlib.lines import Line2D
import pandas as pd
from io import StringIO
import webbrowser
from tempfile import NamedTemporaryFile
import scipy as sp
import pickle
from pureples.hyperneat import create_phenotype_network
from pureples.shared import Substrate, run_hyper


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
                            r'C:\Users\micha\PycharmProjects\Honours Project\NEATHex\config-cppn')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     r'C:\Users\micha\PycharmProjects\Honours Project\NEATHex\config-feedforward')

def evaluate_gaitNEAT(x, duration=5, failed_legs = []):
    net = neat.nn.FeedForwardNetwork.create(x, config)
    # Reset net
    print(failed_legs)
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
    # print(difference)
    # fitness = difference
    # Assign fitness to genome
    x.fitness = fitness
    return fitness


def evaluate_gaitHyperNEAT(x, duration=5, failed_legs = []):
    cppn = neat.nn.FeedForwardNetwork.create(x, CONFIG)
    # Create ANN from CPPN and Substrate
    net = create_phenotype_network(cppn, SUBSTRATE)
    # Reset net

    leg_params = np.array(stationary).reshape(6, 5)
    try:
        controller = HController(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
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


def NEATVHyperNEATStats():
    dfMainBest = pd.DataFrame(columns=['Best NEAT 0', 'Best NEAT 1', 'Best NEAT 2', 'Best NEAT 3', 'Best NEAT 4',
                                       'Best NEAT 5', 'Best NEAT 6', 'Best NEAT 7', 'Best NEAT 8', 'Best NEAT 9',
                                       'Best NEAT 10', 'Best NEAT 11', 'Best NEAT 12', 'Best NEAT 13', 'Best NEAT 14',
                                       'Best NEAT 15', 'Best NEAT 16', 'Best NEAT 17', 'Best NEAT 18',
                                       'Best NEAT 19',
                                       'Best HyperNEAT 0', 'Best HyperNEAT 1', 'Best HyperNEAT 2', 'Best HyperNEAT 3', 'Best HyperNEAT 4',
                                       'Best HyperNEAT 5', 'Best HyperNEAT 6', 'Best HyperNEAT 7', 'Best HyperNEAT 8', 'Best HyperNEAT 9',
                                       'Best HyperNEAT 10', 'Best HyperNEAT 11', 'Best HyperNEAT 12', 'Best HyperNEAT 13', 'Best HyperNEAT 14',
                                       'Best HyperNEAT 15', 'Best HyperNEAT 16', 'Best HyperNEAT 17', 'Best HyperNEAT 18',
                                       'Best HyperNEAT 19',
                                       ])

    for i in range(0, 20):
        NEATfilename = r'C:\Users\micha\PycharmProjects\Honours Project\NEATOutput\genomeFitness\NEATFitnessHistory' + str(
            i) + '.csv'
        HyperNEATfilename = r'C:\Users\micha\PycharmProjects\Honours Project\HyperNEATOutput\genomeFitness\HyperNEATFitnessHistory' + str(
            i) + '.csv'
        dfNEAT = pd.read_csv(NEATfilename)
        dfHyperNEAT = pd.read_csv(HyperNEATfilename)
        dfNEAT.columns = ['Best NEAT', 'Average NEAT']
        dfHyperNEAT.columns = ['Best HyperNEAT', 'Average HyperNEAT']
        dfMainBest['Best NEAT ' + str(i)] = dfNEAT['Best NEAT']
        dfMainBest['Best HyperNEAT ' + str(i)] = dfHyperNEAT['Best HyperNEAT']
        # dfMain['Average NEAT'] = dfMain['Average NEAT'] + df['Average NEAT']
    dfMainBest.reset_index()

    for i in range(1999):
        NEATvals = dfMainBest.iloc[i, 0:20]
        HyperNEATvals = dfMainBest.iloc[i, 20:40]
        print(i)
        print(sp.stats.ranksums(NEATvals, HyperNEATvals, 'greater'))


def mapElitesStats():
    dfNEAT20k = pd.read_csv('NEAT20kQualityMetrics.csv')
    dfNEAT40k = pd.read_csv('NEAT40kQualityMetrics.csv')
    dfHyperNEAT20k = pd.read_csv('HyperNEAT20kQualityMetrics.csv')
    dfHyperNEAT40k = pd.read_csv('HyperNEAT40kQualityMetrics.csv')

    dfNEAT20kMax = dfNEAT20k.max(axis=0)
    dfNEAT40kMax = dfNEAT40k.max(axis=0)
    dfHyperNEAT20kMax = dfHyperNEAT20k.max(axis=0)
    dfHyperNEAT40kMax = dfHyperNEAT40k.max(axis=0)

    dfNEAT20kAverage = dfNEAT20k.mean(axis=0).to_numpy()
    print(type(dfNEAT20kAverage))
    dfNEAT40kAverage = dfNEAT40k.mean(axis=0).to_numpy()
    dfHyperNEAT20kAverage = dfHyperNEAT20k.mean(axis=0).to_numpy()
    dfHyperNEAT40kAverage = dfHyperNEAT40k.mean(axis=0).to_numpy()

    rng = np.random.default_rng()
    sample = rng.uniform(0, 20, 300)

    ## Rank Sums tests on global reliability
    # NEAT 20k vs 40k
    print(sp.stats.wilcoxon(dfNEAT20kAverage, dfNEAT40kAverage, alternative='greater'))
    # HyperNEAT 20k vs 40k
    print(sp.stats.wilcoxon(dfHyperNEAT20kAverage, dfHyperNEAT40kAverage, alternative='greater'))
    # NEAT 20k vs HyperNEAT 20k
    print(sp.stats.wilcoxon(dfNEAT20kAverage, dfHyperNEAT20kAverage, alternative='greater'))
    # NEAT 40k vs HyperNEAT 40k
    print(sp.stats.wilcoxon(dfNEAT40kAverage, dfHyperNEAT40kAverage, alternative='greater'))

def MBOAStats():
    ## Calculate reference gait performance
    # Get NEAT reference Gait
    filename = r'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_20000archive\archive_genome8011476.pkl'
    with open(filename, 'rb') as f:
        genomes = pickle.load(f)
    genome = genomes[225]

    # Get HyperNEAT reference Gait
    filename = r'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\3_20000archive\archive_genome8001878.pkl'
    with open(filename, 'rb') as f:
        genomes = pickle.load(f)
    genome2 = genomes[141]

    S0 = [[]]
    S1 = [[1], [2], [3], [4], [5], [6]]
    S2 = [[1, 4], [2, 5], [3, 6]]
    S3 = [[1, 3], [2, 4], [3, 5], [4, 6], [5, 1], [6, 2]]
    S4 = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]]

    scenarios = [S0, S1, S2, S3, S4]
    NEATr0 = []
    NEATr1 = []
    NEATr2 = []
    NEATr3 = []
    NEATr4 = []
    NEATReferenceResults = [NEATr0, NEATr1, NEATr2, NEATr3, NEATr4]

    HyperNEATr0 = []
    HyperNEATr1 = []
    HyperNEATr2 = []
    HyperNEATr3 = []
    HyperNEATr4 = []
    HyperNEATReferenceResults = [HyperNEATr0, HyperNEATr1, HyperNEATr2, HyperNEATr3, HyperNEATr4]
    index = 0
    for scenario in scenarios:
        for s in scenario:
            performanceNEAT = evaluate_gaitNEAT(genome, failed_legs=s)
            NEATReferenceResults[index].append(performanceNEAT)
            performanceHyperNEAT = evaluate_gaitHyperNEAT(genome2, failed_legs=s)
            HyperNEATReferenceResults[index].append(performanceHyperNEAT)
        index += 1

    NEAT20SList = []
    for i in range(5):
        CurrentNEAT = np.loadtxt(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEATSim\20000_niches\perfs_{i}.dat').flatten()
        NEAT20SList.append(CurrentNEAT)

    NEAT40SList = []
    for i in range(5):
        CurrentNEAT = np.loadtxt(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEATSim\40000_niches\perfs_{i}.dat').flatten()
        NEAT40SList.append(CurrentNEAT)

    HyperNEAT20SList = []
    for i in range(5):
        CurrentHyperNEAT = np.loadtxt(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEATSim\20000_niches\perfs_{i}.dat').flatten()
        HyperNEAT20SList.append(CurrentHyperNEAT)

    HyperNEAT40SList = []
    for i in range(5):
        CurrentHyperNEAT = np.loadtxt(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEATSim\40000_niches\perfs_{i}.dat').flatten()
        HyperNEAT40SList.append(CurrentHyperNEAT)

    compDF = pd.DataFrame(columns=['N20N', 'N20H', 'N40N', 'N40H', 'H20N', 'H20H', 'H40N', 'H40H'])
    for i in range(1, 5):
        #print("NEAT20k vs NEAT Reference S" + str(i))
        NEAT20S = NEAT20SList[i]
        compDF.loc[i, 'N20N'] = sp.stats.ranksums(NEAT20S,  NEATReferenceResults[i], 'greater')[1]
        print(type(sp.stats.ranksums(NEAT20S, NEATReferenceResults[i], 'greater')[1]))
        print("NEAT20k vs HyperNEAT Reference S" + str(i))
        compDF.loc[i, 'N20H'] = sp.stats.ranksums(NEAT20S, HyperNEATReferenceResults[i], 'greater')[1]

        print("NEAT40k vs NEAT Reference S" + str(i))
        NEAT40S = NEAT40SList[i]
        compDF.loc[i, 'N40N'] = sp.stats.ranksums(NEAT40S, NEATReferenceResults[i], 'greater')[1]
        print("NEAT40k vs Reference S" + str(i))
        compDF.loc[i, 'N40H'] = sp.stats.ranksums(NEAT40S, HyperNEATReferenceResults[i], 'greater')[1]

        print("HyperNEAT20k vs NEAT Reference S" + str(i))
        HyperNEAT20S = HyperNEAT20SList[i]
        compDF.loc[i, 'H20N'] = sp.stats.ranksums(HyperNEAT20S, NEATReferenceResults[i], 'greater')[1]
        print("HyperNEAT20k vs HyperNEAT Reference S" + str(i))
        compDF.loc[i, 'H20H'] = sp.stats.ranksums(HyperNEAT20S, HyperNEATReferenceResults[i], 'greater')[1]

        print("HyperNEAT40k vs NEAT Reference S" + str(i))
        HyperNEAT40S = HyperNEAT40SList[i]
        compDF.loc[i, 'H40N'] = sp.stats.ranksums(HyperNEAT40S, NEATReferenceResults[i], 'greater')[1]
        print("HyperNEAT40k vs HyperNEAT Reference S" + str(i))
        compDF.loc[i, 'H40H'] = sp.stats.ranksums(HyperNEAT40S, HyperNEATReferenceResults[i], 'greater')[1]
    compDF.to_csv('MBOA p-values.csv')



NEATVHyperNEATStats()
#mapElitesStats()
#MBOAStats()
