import warnings
from hexapod.controllers.NEATController import Controller, tripod_gait, reshape, stationary
from hexapod.simulator import Simulator
from hexapod.controllers.anglequee import  anglequee
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import time
import neat
import pickle

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     r'C:\Users\micha\PycharmProjects\Honours Project\NEATHex\config-feedforward')

def average_fitness():
    start_time = time.time()
    dfMain = pd.read_csv(r'C:\Users\micha\PycharmProjects\Honours Project\NEATOutput\genomeFitness\NEATFitnessHistory0.csv')
    dfMain.columns = ['Best NEAT', 'Average NEAT']
    dfMain['Lowest NEAT'] = dfMain['Best NEAT']
    dfMain['Highest NEAT'] = dfMain['Best NEAT']
    dfMainLowestHighest = pd.DataFrame(columns=['Lowest NEAT', 'Highest NEAT', 'NEAT current'])
    dfMainLowestHighest['Lowest NEAT'] = dfMain['Best NEAT']
    dfMainLowestHighest['Highest NEAT'] = dfMain['Best NEAT']
    for i in range(1, 20):
        filename = r'C:\Users\micha\PycharmProjects\Honours Project\NEATOutput\genomeFitness\NEATFitnessHistory' + str(i) + '.csv'
        df = pd.read_csv(filename)
        df.columns = ['Best NEAT', 'Average NEAT']
        dfMain['Best NEAT'] = dfMain['Best NEAT'] + df['Best NEAT']
        dfMain['Average NEAT'] = dfMain['Average NEAT'] + df['Average NEAT']
        dfMainLowestHighest['NEAT current'] = df['Best NEAT']
        dfMainLowestHighest['Highest NEAT'] = dfMainLowestHighest.max(axis=1)
        dfMainLowestHighest['Lowest NEAT'] = dfMainLowestHighest.min(axis=1)
    dfMain['Best NEAT'] = dfMain['Best NEAT'] / 20
    dfMain['Average NEAT'] = dfMain['Average NEAT'] / 20
    maximum = dfMain['Best NEAT'].max()
    dfMain['Best NEAT'] = dfMain['Best NEAT'] / maximum
    dfMain['Average NEAT'] = dfMain['Average NEAT'] / maximum
    dfMain.reset_index()

    dfMain2 = pd.read_csv(
        r'C:\Users\micha\PycharmProjects\Honours Project\HyperNEATOutput\genomeFitness\HyperNEATFitnessHistory0.csv')
    dfMain2.columns = ['Best HyperNEAT', 'Average HyperNEAT']
    dfMain2LowestHighest = pd.DataFrame(columns=['Lowest HyperNEAT', 'Highest HyperNEAT', 'HyperNEAT current'])
    dfMain2LowestHighest['Lowest HyperNEAT'] = dfMain2['Best HyperNEAT']
    dfMain2LowestHighest['Highest HyperNEAT'] = dfMain2['Best HyperNEAT']
    for i in range(1, 20):
        filename = r'C:\Users\micha\PycharmProjects\Honours Project\HyperNEATOutput\genomeFitness\HyperNEATFitnessHistory' + str(
            i) + '.csv'
        df = pd.read_csv(filename)
        df.columns = ['Best HyperNEAT', 'Average HyperNEAT']
        dfMain2['Best HyperNEAT'] = dfMain2['Best HyperNEAT'] + df['Best HyperNEAT']
        dfMain2['Average HyperNEAT'] = dfMain2['Average HyperNEAT'] + df['Average HyperNEAT']
        dfMain2LowestHighest['HyperNEAT current'] = df['Best HyperNEAT']
        dfMain2LowestHighest['Highest HyperNEAT'] = dfMain2LowestHighest.max(axis=1)
        dfMain2LowestHighest['Lowest HyperNEAT'] = dfMain2LowestHighest.min(axis=1)
    dfMain2['Best HyperNEAT'] = dfMain2['Best HyperNEAT'] / 20 / maximum
    dfMain2['Average HyperNEAT'] = dfMain2['Average HyperNEAT'] / 20 / maximum
    dfMain2.reset_index()

    dfMain = pd.concat([dfMain, dfMain2], axis=1)
    plt.plot(dfMain.index, dfMain['Best NEAT'])
    #plt.fill_between(dfMain.index, dfMainLowestHighest['Lowest NEAT'], dfMainLowestHighest['Highest NEAT'], alpha = 0.2)
    plt.plot(dfMain.index, dfMain['Best HyperNEAT'])
    #plt.fill_between(dfMain.index, dfMain2LowestHighest['Lowest HyperNEAT'], dfMain2LowestHighest['Highest HyperNEAT'], alpha=0.2)
    plt.plot(dfMain.index, dfMain['Average NEAT'])
    plt.plot(dfMain.index, dfMain['Average HyperNEAT'])
    plt.title("Fitness NEAT vs HyperNEAT")
    plt.legend(['Best NEAT', 'Best HyperNEAT', 'Average NEAT', 'Average HyperNEAT'], loc = 'upper left')
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness")
    #plt.ylim([0,1.1])
    plt.savefig("NEAT v HyperNEAT.png")
    plt.show()

    # lines = dfMain.plot.line()
    # plt.xlabel("Number of Generations")
    # plt.ylabel("Fitness")
    # plt.savefig("NEATVHyperNEAT")
    # plt.show()
    # plt.close()

def legCoordinationNEAT(NUM_SECONDS, filename):
    with open(filename, 'rb') as f:
        winner = pickle.load(f)
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    controller = Controller(tripod_gait, body_height=0.15, velocity=0.5, crab_angle=-np.pi / 6, ann=winner_net,
                            printangles=True)
    simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=[0])

    log = []
    log_sorted = {11: [], 12: [], 13: [], 21: [], 22: [], 23: [], 31: [], 32: [], 33: [], 41: [], 42: [], 43: [],
                  51: [], 52: [], 53: [], 61: [], 62: [], 63: []}
    x = 0

    aq = anglequee()
    while x < (240*NUM_SECONDS)-1:
        simulator.step()
        time.sleep(0.03)

        current_array = simulator.controller.joint_angles(x/240.0)

        log_sorted[11].append(current_array[1])
        log_sorted[12].append(current_array[1])
        log_sorted[13].append(current_array[2])
        log_sorted[21].append(current_array[3])
        log_sorted[22].append(current_array[4])
        log_sorted[23].append(current_array[5])
        log_sorted[31].append(current_array[6])
        log_sorted[32].append(current_array[7])
        log_sorted[33].append(current_array[8])
        log_sorted[41].append(current_array[9])
        log_sorted[42].append(current_array[10])
        log_sorted[43].append(current_array[11])
        log_sorted[51].append(current_array[12])
        log_sorted[52].append(current_array[13])
        log_sorted[53].append(current_array[14])
        log_sorted[61].append(current_array[15])
        log_sorted[62].append(current_array[16])
        log_sorted[63].append(current_array[17])
        x = x+1

    t=np.linspace(0, NUM_SECONDS, (240*NUM_SECONDS)-1)

    plt.plot(t, log_sorted[11], label='joint 1')
    # plt.plot(t, log_sorted[12], label='joint 2')
    # plt.plot(t, log_sorted[13], label='joint 3')
    plt.plot(t, log_sorted[21], label='joint 4')
    # plt.plot(t, log_sorted[22], label='joint 5')
    # plt.plot(t, log_sorted[23], label='joint 6')
    plt.plot(t, log_sorted[31], label='joint 7')
    # plt.plot(t, log_sorted[32], label='joint 8')
    # plt.plot(t, log_sorted[33], label='joint 9')
    plt.plot(t, log_sorted[41], label='joint 10')
    # plt.plot(t, log_sorted[42], label='joint 11')
    # plt.plot(t, log_sorted[43], label='joint 12')
    plt.plot(t, log_sorted[51], label='joint 13')
    # plt.plot(t, log_sorted[52], label='joint 14')
    # plt.plot(t, log_sorted[53], label='joint 15')
    plt.plot(t, log_sorted[61], label='joint 16')
    # plt.plot(t, log_sorted[62], label='joint 17')
    # plt.plot(t, log_sorted[62], label='joint 18')

    plt.xlabel("Time in Seconds")
    plt.ylabel("Joint angles")
    plt.title("Controller joint angles")
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    average_fitness()
    filename = r"C:\Users\micha\PycharmProjects\Honours Project\NEATOutput\bestGenomes\NEATGenome1.pkl"
   # legCoordinationNEAT(NUM_SECONDS=1, filename = filename)
