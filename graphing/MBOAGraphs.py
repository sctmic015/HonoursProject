from hexapod.controllers.NEATController import Controller, tripod_gait, reshape, stationary
from hexapod.controllers.hyperNEATController import Controller as HController
from hexapod.simulator import Simulator
import neat
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats
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

def adaptive_performance_plots(tripod_mean_dataNEAT, tripod_mean_dataHyperNEAT):
    # matplotlib.rcParams.update({
    #     'pgf.texsystem': "pdflatex",
    #     'pdf.fonttype': 42,
    #     'ps.fonttype': 42,
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color='black')

    sim_data_NEAT20 = []
    sim_data_NEAT40 = []
    sim_data_HyperNEAT20 = []
    sim_data_HyperNEAT40 = []

    for scenario in range(5):
        data_NEAT20 = list(np.loadtxt(
            rf"C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEATSim\20000_niches\perfs_{scenario}.dat").flatten())
        data_NEAT40 = list(np.loadtxt(
            rf"C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEATSim\40000_niches\perfs_{scenario}.dat").flatten())
        sim_data_NEAT20.append(data_NEAT20)
        sim_data_NEAT40.append(data_NEAT40)
        data_HyperNEAT20 = list(np.loadtxt(
            rf"C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEATSim\20000_niches\perfs_{scenario}.dat").flatten())
        data_HyperNEAT40 = list(np.loadtxt(
            rf"C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEATSim\40000_niches\perfs_{scenario}.dat").flatten())
        sim_data_HyperNEAT20.append(data_HyperNEAT20)
        sim_data_HyperNEAT40.append(data_HyperNEAT40)

    # performance stats. Not required for graph. Should have a reference perhaps
    # print("performance stats")
    # for scenario in range(1, 5):
    #     t_statistic, p_value = stats.ttest_ind(sim_data_NEAT20[scenario] + sim_data_NEAT40[scenario], tripod_data[scenario])
    #     print(f'S{scenario}:', p_value)

    # map size stats. Also not really needed for graphs
    print('map size statistics')
    for scenario in range(4):
        t_statistic, p_value = stats.ttest_ind(sim_data_NEAT20[scenario], sim_data_NEAT40[scenario])
        print(f'S{scenario + 1}:', p_value)

    ticks = ['None', 'S1', 'S2', 'S3', 'S4']
    box_width = 0.4
    color_NEAT20 = 'tab:orange'
    color_NEAT40 = 'tab:blue'
    color_HyperNEAT20 = 'tab:green'
    color_HyperNEAT40 = 'tab:red'

    fig, ax = plt.subplots()
    #fig.set_size_inches(w=3.3, h=2.0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.set_title('Adapted Fitness')
    ax.set_xlabel('Failure scenario')
    ax.set_ylabel('Fitness (m)')

    flierprops = dict(marker='o', markersize=5, linestyle='none', markeredgecolor='darkgray')
    meanline = dict(linestyle='-', color='black')
    meanpoint = dict(marker='D', markeredgecolor='black', markerfacecolor='red')
    positions = np.array(range(len(sim_data_NEAT40))) * 4.0      ## Maybe change to 4

    bpNEAT20 = plt.boxplot(sim_data_NEAT20, positions=positions - 0.75, widths=box_width, showfliers=True,
                       flierprops=flierprops, patch_artist=True)
    bpNEAT40 = plt.boxplot(sim_data_NEAT40, positions=positions - 0.25, widths=box_width, showfliers=True,
                       flierprops=flierprops, patch_artist=True)
    bpHyperNEAT20 = plt.boxplot(sim_data_HyperNEAT20, positions=positions + 0.75, widths=box_width, showfliers=True,
                       flierprops=flierprops, patch_artist=True)
    bpHyperNEAT40 = plt.boxplot(sim_data_HyperNEAT40, positions=positions + 1.25, widths=box_width, showfliers=True,
                                 flierprops=flierprops, patch_artist=True)

    plt.scatter(positions, tripod_mean_dataNEAT, marker='*', color='tab:purple')   # Might use with ref gaits of some sort
    plt.scatter(positions, tripod_mean_dataHyperNEAT, marker='*', color='tab:pink')

    set_box_color(bpNEAT20, color_NEAT20)
    set_box_color(bpNEAT40, color_NEAT40)
    set_box_color(bpHyperNEAT20, color_HyperNEAT20)
    set_box_color(bpHyperNEAT40, color_HyperNEAT40)

    custom_lines = [mpatches.Patch(color=color_NEAT20), mpatches.Patch(color=color_NEAT40), mpatches.Patch(color=color_HyperNEAT20), mpatches.Patch(color=color_HyperNEAT40)] ## DOn't forget hyperNEAT40
    plt.legend(custom_lines, ['NEAT 20k', 'NEAT 40k', 'HyperNEAT 20k', 'HyperNEAT 40k'])   # Don't forget

    plt.xticks(range(0, len(ticks) * 4, 4), ticks)
    plt.xlim(-4, len(ticks) * 4)
    #plt.tight_layout(pad=0.1)
    plt.savefig("Adaption Fitness.png")
    plt.show()

def number_of_adaptations_plot():
    # matplotlib.rcParams.update({
    #     'pgf.texsystem': "pdflatex",
    #     'pdf.fonttype': 42,
    #     'ps.fonttype': 42,
    #     'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color='black')

    sim_data_NEAT20 = []
    sim_data_NEAT40 = []
    sim_data_HyperNEAT20 = []
    sim_data_HyperNEAT40 = []

    for scenario in range(1, 5):
        sim_data_NEAT20.append(list(np.loadtxt(rf"C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEATSim\20000_niches\trials_{scenario}.dat").flatten()))
        sim_data_NEAT40.append(list(np.loadtxt(rf"C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEATSim\40000_niches\trials_{scenario}.dat").flatten()))
        sim_data_HyperNEAT20.append(list(np.loadtxt(rf"C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEATSim\20000_niches\trials_{scenario}.dat").flatten()))
        sim_data_HyperNEAT40.append(list(np.loadtxt(rf"C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEATSim\40000_niches\trials_{scenario}.dat").flatten()))

    # adaptation stats
    print('adaptation statistics')

    # map size stats
    print('map size statistics')
    t_statistic, p_value = stats.ttest_ind(sim_data_NEAT20[0], sim_data_NEAT40[0])
    print('S1:', p_value)
    t_statistic, p_value = stats.ttest_ind(sim_data_NEAT20[1], sim_data_NEAT40[1])
    print('S2:', p_value)
    t_statistic, p_value = stats.ttest_ind(sim_data_NEAT20[2], sim_data_NEAT40[2])
    print('S3:', p_value)
    t_statistic, p_value = stats.ttest_ind(sim_data_NEAT20[3], sim_data_NEAT40[3])
    print('S4:', p_value)

    ticks = ['S1', 'S2', 'S3', 'S4']
    color_NEAT20 = 'tab:orange'
    color_NEAT40 = 'tab:blue'
    color_HyperNEAT20 = 'tab:green'
    color_HyperNEAT40 = 'tab:red'
    box_width = 0.4

    fig, ax = plt.subplots()
    #fig.set_size_inches(w=3.3, h=2.0)
    ax.yaxis.grid(True)
    ax.set_title('Number of Adaptation Trials')
    ax.set_xlabel('Failure scenario')
    ax.set_ylabel('Trials')

    flierprops = dict(marker='o', markersize=5, linestyle='none', markeredgecolor='darkgray')
    positions = np.array(range(len(sim_data_NEAT40))) * 4.0
    bpNEAT20 = plt.boxplot(sim_data_NEAT20, positions=positions - 0.85, widths=box_width, showfliers=True,
                       flierprops=flierprops, patch_artist=True)
    bpNEAT40 = plt.boxplot(sim_data_NEAT40, positions=positions - 0.35, widths=box_width, showfliers=True,
                       flierprops=flierprops, patch_artist=True)
    bpHyperNEAT20 = plt.boxplot(sim_data_HyperNEAT20, positions=positions + 0.35, widths=box_width, showfliers=True,
                       flierprops=flierprops, patch_artist=True)
    bpHyperNEAT40 = plt.boxplot(sim_data_HyperNEAT40, positions=positions + 0.85, widths=box_width, showfliers=True,
                        flierprops=flierprops, patch_artist=True)
    set_box_color(bpNEAT20, color_NEAT20)
    set_box_color(bpNEAT40, color_NEAT40)
    set_box_color(bpHyperNEAT20, color_HyperNEAT20)
    set_box_color(bpHyperNEAT40, color_HyperNEAT40)

    custom_lines = [mpatches.Patch(color=color_NEAT20), mpatches.Patch(color=color_NEAT40), mpatches.Patch(color=color_HyperNEAT20), mpatches.Patch(color=color_HyperNEAT40)]
    plt.legend(custom_lines, ['20k NEAT', '40k NEAT', '20k HyperNEAT', '40k HyperNEAT'])

    plt.xticks(range(0, len(ticks) * 4, 4), ticks)
    plt.xlim(-4, len(ticks) * 4)
    plt.ylim(0, 35)
    #plt.tight_layout(pad=0.1)

    plt.savefig("adaptation trials.png")
    plt.show()

if __name__ == '__main__':
    # Calculate NEAT reference Gait
    filename = r'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_20000archive\archive_genome8011476.pkl'
    with open(filename, 'rb') as f:
        genomes = pickle.load(f)
    genome = genomes[225]

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
    tripod_mean_NEAT = np.array([])
    tripod_mean_HyperNEAT = np.array([])
    for scenario in scenarios:
        currentMeanNEAT = 0
        currentMeanHyperNEAT = 0
        currentCount = 0
        for s in scenario:
            performanceNEAT = evaluate_gaitNEAT(genome, failed_legs=s)
            performanceHyperNEAT = evaluate_gaitHyperNEAT(genome2, failed_legs=s)
            currentMeanNEAT += performanceNEAT
            currentMeanHyperNEAT += performanceHyperNEAT
            currentCount += 1
        currentMeanNEAT = currentMeanNEAT / currentCount
        tripod_mean_NEAT = np.append(tripod_mean_NEAT, currentMeanNEAT)
        currentMeanHyperNEAT = currentMeanHyperNEAT / currentCount
        tripod_mean_HyperNEAT = np.append(tripod_mean_HyperNEAT, currentMeanHyperNEAT)


    adaptive_performance_plots(tripod_mean_dataNEAT=tripod_mean_NEAT, tripod_mean_dataHyperNEAT=tripod_mean_HyperNEAT)
    number_of_adaptations_plot()


