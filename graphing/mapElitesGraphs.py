# | Author: Michael Scott

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from matplotlib.lines import Line2D
import pandas as pd
from io import StringIO
import webbrowser
from tempfile import NamedTemporaryFile


def progression_metrics():

    ## Load in Data for NEAT 20k
    dfMainNEAT = pd.read_csv(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_20000\log.dat',
                             header=None, sep='\s', engine='python')
    dfMainNEAT.columns = ['Evaluations', 'Niches', 'Best NEAT 20k', 'Average NEAT 20k', 'Median NEAT 20k',
                          '5th percentile', '95th percentile']
    dfMainNEAT = dfMainNEAT.drop(dfMainNEAT.index[3349:len(dfMainNEAT)])
    dfMainNEATBestLine = pd.DataFrame(columns=['Best NEAT 20k'])
    dfMainNEATBestLine['Best NEAT 20k'] = dfMainNEAT['Best NEAT 20k']
    dfMainNEATBestLineAverage = pd.DataFrame(columns=['Average NEAT 20k'])
    dfMainNEATBestLineAverage['Average NEAT 20k'] = dfMainNEAT['Average NEAT 20k']
    for i in range(1, 10):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\{i}_20000\log.dat'
        df = pd.read_csv(filename, header=None, sep='\s', engine='python')
        df.columns = ['Evaluations', 'Niches', 'Best NEAT 20k', 'Average NEAT 20k', 'Median NEAT 20k', '5th percentile',
                      '95th percentile']
        dfMainNEAT = dfMainNEAT + df
        dfMainNEATBestLine.insert(i, 'Best NEAT 20k ' + str(i), df['Best NEAT 20k'], True)
        dfMainNEATBestLineAverage.insert(i, 'Average NEAT 20k' + str(i), df['Average NEAT 20k'], True)
        # dfMainNEATBestLine['Best NEAT 20k ' + str(i)] = df['Best NEAT 20k']
    dfMainNEATBestLine.insert(10, 'Highest NEAT 20k', dfMainNEATBestLine.max(axis=1), True)
    dfMainNEATBestLine.insert(10, 'Lowest NEAT 20k', dfMainNEATBestLine.min(axis=1), True)
    dfMainNEATBestLineAverage.insert(10, 'Highest Average NEAT 20k', dfMainNEATBestLineAverage.max(axis=1), True)
    dfMainNEATBestLineAverage.insert(10, 'Lowest Average NEAT 20k', dfMainNEATBestLineAverage.min(axis=1), True)
    dfMainNEAT = dfMainNEAT / 10
    dfMainNEAT.reset_index()


    ## Load in Data HyperNEAT 20k
    dfMainHyperNEAT = pd.read_csv(
        rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\0_20000\log.dat',
        header=None, sep='\s', engine='python')
    dfMainHyperNEAT.columns = ['Evaluations', 'Niches', 'Best HyperNEAT 20k', 'Average HyperNEAT 20k',
                               'Median HyperNEAT 20k',
                               '5th percentile', '95th percentile']
    dfMainNEATHyperNEAT = dfMainHyperNEAT.drop(dfMainHyperNEAT.index[3349:len(dfMainHyperNEAT)])
    dfMainHyperNEATBestLine = pd.DataFrame(columns=['Best Hyper NEAT 20k'])
    dfMainHyperNEATBestLine['Best HyperNEAT 20k'] = dfMainHyperNEAT['Best HyperNEAT 20k']
    dfMainHyperNEATBestLineAverage = pd.DataFrame(columns=['Average HyperNEAT 20k'])
    dfMainHyperNEATBestLineAverage['Average HyperNEAT 20k'] = dfMainHyperNEAT['Average HyperNEAT 20k']
    for i in range(1, 5):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\{i}_20000\log.dat'
        df = pd.read_csv(filename, header=None, sep='\s', engine='python')
        df.columns = ['Evaluations', 'Niches', 'Best HyperNEAT 20k', 'Average HyperNEAT 20k', 'Median HyperNEAT 20k',
                      '5th percentile',
                      '95th percentile']
        dfMainHyperNEAT = dfMainHyperNEAT + df
        dfMainHyperNEATBestLine.insert(i, 'Best HyperNEAT 20k ' + str(i), df['Best HyperNEAT 20k'], True)
        dfMainHyperNEATBestLineAverage.insert(i, 'Average HyperNEAT 20k' + str(i), df['Average HyperNEAT 20k'], True)
        # dfMainNEATBestLine['Best NEAT 20k ' + str(i)] = df['Best NEAT 20k']
    dfMainHyperNEATBestLine.insert(5, 'Highest HyperNEAT 20k', dfMainHyperNEATBestLine.max(axis=1), True)
    dfMainHyperNEATBestLine.insert(5, 'Lowest HyperNEAT 20k', dfMainHyperNEATBestLine.min(axis=1), True)
    dfMainHyperNEATBestLineAverage.insert(5, 'Highest Average HyperNEAT 20k',
                                          dfMainHyperNEATBestLineAverage.max(axis=1), True)
    dfMainHyperNEATBestLineAverage.insert(5, 'Lowest Average HyperNEAT 20k', dfMainHyperNEATBestLineAverage.min(axis=1),
                                          True)
    dfMainHyperNEAT = dfMainHyperNEAT / 5
    dfMainHyperNEAT.reset_index()

    ## Load in Data for NEAT 40k
    dfMainNEAT2 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_40000\log.dat',
                              header=None, sep='\s', engine='python')
    dfMainNEAT2.columns = ['Evaluations', 'Niches', 'Best NEAT 40k', 'Average NEAT 40k', 'Median NEAT 40k',
                           '5th percentile', '95th percentile']
    dfMainNEAT2 = dfMainNEAT2.drop(dfMainNEAT2.index[3349:len(dfMainNEAT)])
    dfMainNEATBestLine2 = pd.DataFrame(columns=['Best NEAT 40k'])
    dfMainNEATBestLine2['Best NEAT 40k'] = dfMainNEAT2['Best NEAT 40k']
    dfMainNEATBestLineAverage2 = pd.DataFrame(columns=['Average NEAT 40k'])
    dfMainNEATBestLineAverage2['Average NEAT 40k'] = dfMainNEAT2['Average NEAT 40k']
    for i in range(1, 10):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\{i}_40000\log.dat'
        df = pd.read_csv(filename, header=None, sep='\s', engine='python')
        df.columns = ['Evaluations', 'Niches', 'Best NEAT 40k', 'Average NEAT 40k', 'Median NEAT 40k', '5th percentile',
                      '95th percentile']
        dfMainNEAT2 = dfMainNEAT2 + df
        dfMainNEATBestLine2.insert(i, 'Best NEAT 40k ' + str(i), df['Best NEAT 40k'], True)
        dfMainNEATBestLineAverage2.insert(i, 'Average NEAT 40k' + str(i), df['Average NEAT 40k'], True)
    dfMainNEATBestLine2.insert(10, 'Highest NEAT 40k', dfMainNEATBestLine2.max(axis=1), True)
    dfMainNEATBestLine2.insert(10, 'Lowest NEAT 40k', dfMainNEATBestLine2.min(axis=1), True)
    dfMainNEATBestLineAverage2.insert(10, 'Highest Average NEAT 40k', dfMainNEATBestLineAverage2.max(axis=1), True)
    dfMainNEATBestLineAverage2.insert(10, 'Lowest Average NEAT 40k', dfMainNEATBestLineAverage2.min(axis=1), True)
    dfMainNEAT2 = dfMainNEAT2 / 10
    dfMainNEAT2.reset_index()

    ## Load in Data for HyperNEAT 40k
    dfMainHyperNEAT2 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\0_40000\log.dat',
                              header=None, sep='\s', engine='python')
    dfMainHyperNEAT2.columns = ['Evaluations', 'Niches', 'Best HyperNEAT 40k', 'Average HyperNEAT 40k', 'Median HyperNEAT 40k',
                           '5th percentile', '95th percentile']
    dfMainHyperNEAT2 = dfMainHyperNEAT2.drop(dfMainHyperNEAT2.index[3349:len(dfMainHyperNEAT)])
    dfMainHyperNEATBestLine2 = pd.DataFrame(columns=['Best HyperNEAT 40k'])
    dfMainHyperNEATBestLine2['Best HyperNEAT 40k'] = dfMainHyperNEAT2['Best HyperNEAT 40k']
    dfMainHyperNEATBestLineAverage2 = pd.DataFrame(columns=['Average HyperNEAT 40k'])
    dfMainHyperNEATBestLineAverage2['Average HyperNEAT 40k'] = dfMainHyperNEAT2['Average HyperNEAT 40k']
    for i in range(1, 5):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\{i}_40000\log.dat'
        df = pd.read_csv(filename, header=None, sep='\s', engine='python')
        df.columns = ['Evaluations', 'Niches', 'Best HyperNEAT 40k', 'Average HyperNEAT 40k', 'Median HyperNEAT 40k', '5th percentile',
                      '95th percentile']
        dfMainHyperNEAT2 = dfMainHyperNEAT2 + df
        dfMainHyperNEATBestLine2.insert(i, 'Best HyperNEAT 40k ' + str(i), df['Best HyperNEAT 40k'], True)
        dfMainHyperNEATBestLineAverage2.insert(i, 'Average HyperNEAT 40k' + str(i), df['Average HyperNEAT 40k'], True)
    dfMainHyperNEATBestLine2.insert(5, 'Highest HyperNEAT 40k', dfMainHyperNEATBestLine2.max(axis=1), True)
    dfMainHyperNEATBestLine2.insert(5, 'Lowest HyperNEAT 40k', dfMainHyperNEATBestLine2.min(axis=1), True)
    dfMainHyperNEATBestLineAverage2.insert(5, 'Highest Average HyperNEAT 40k', dfMainHyperNEATBestLineAverage2.max(axis=1), True)
    dfMainHyperNEATBestLineAverage2.insert(5, 'Lowest Average HyperNEAT 40k', dfMainHyperNEATBestLineAverage2.min(axis=1), True)
    dfMainHyperNEAT2 = dfMainHyperNEAT2 / 5
    dfMainHyperNEAT2.reset_index()

    ## Find highest fitness to normalise graphs
    referenceFitness = max(dfMainNEATBestLine.max())
    print(referenceFitness)

    ## Collate Statistics for Best Graph
    dfBest = pd.DataFrame(
        columns=['Evaluations', 'Best NEAT 20k', 'Best NEAT 40k', 'Highest NEAT 40k', 'Lowest NEAT 40k'])
    dfBest['Evaluations'] = dfMainNEAT['Evaluations']
    dfBest['Best NEAT 20k'] = dfMainNEAT['Best NEAT 20k'] / referenceFitness
    dfBest['Best NEAT 40k'] = dfMainNEAT2['Best NEAT 40k'] / referenceFitness
    dfBest['Highest NEAT 20k'] = dfMainNEATBestLine['Highest NEAT 20k'] / referenceFitness
    dfBest['Lowest NEAT 20k'] = dfMainNEATBestLine['Lowest NEAT 20k'] / referenceFitness
    dfBest['Highest NEAT 40k'] = dfMainNEATBestLine2['Highest NEAT 40k'] / referenceFitness
    dfBest['Lowest NEAT 40k'] = dfMainNEATBestLine2['Lowest NEAT 40k'] / referenceFitness
    dfBest['Best HyperNEAT 20k'] = dfMainHyperNEAT['Best HyperNEAT 20k'] / referenceFitness
    dfBest['Best HyperNEAT 40k'] = dfMainHyperNEAT2['Best HyperNEAT 40k'] / referenceFitness
    dfBest['Highest HyperNEAT 20k'] = dfMainHyperNEATBestLine['Highest HyperNEAT 20k'] / referenceFitness
    dfBest['Lowest HyperNEAT 20k'] = dfMainHyperNEATBestLine['Lowest HyperNEAT 20k'] / referenceFitness
    dfBest['Highest HyperNEAT 40k'] = dfMainHyperNEATBestLine2['Highest HyperNEAT 40k'] / referenceFitness
    dfBest['Lowest HyperNEAT 40k'] = dfMainHyperNEATBestLine2['Lowest HyperNEAT 40k'] / referenceFitness

    # Plot Best Graph
    plt.plot(dfBest['Evaluations'], dfBest['Best NEAT 20k'])
    plt.fill_between(dfBest['Evaluations'], dfBest['Lowest NEAT 20k'], dfBest['Highest NEAT 20k'] + 0.01, alpha=0.2)
    plt.plot(dfBest['Evaluations'], dfBest['Best NEAT 40k'])
    plt.fill_between(dfBest['Evaluations'], dfBest['Lowest NEAT 40k'], dfBest['Highest NEAT 40k'] + 0.01, alpha=0.2)
    plt.plot(dfBest['Evaluations'], dfBest['Best HyperNEAT 20k'])
    plt.fill_between(dfBest['Evaluations'], dfBest['Lowest HyperNEAT 20k'], dfBest['Highest HyperNEAT 20k'] + 0.01,
                     alpha=0.2)
    plt.plot(dfBest['Evaluations'], dfBest['Best HyperNEAT 40k'])
    plt.fill_between(dfBest['Evaluations'], dfBest['Lowest HyperNEAT 20k'], dfBest['Best HyperNEAT 40k'])
    #plt.ylim(0, 1)
    plt.title('Best Performance')
    plt.ylabel('Best Fitness')
    plt.xlabel('Evaluations ($x10^6$)')
    plt.legend(['NEAT 20k', 'NEAT 40k', 'HyperNEAT 20k', 'HyperNEAT40k'])
    plt.savefig("Maximum Performance.png")
    plt.show()

    ## Collate Statistics for Average Graph
    dfAverage = pd.DataFrame(columns=['Evaluations', 'Average NEAT 20k', 'Average NEAT 40k', 'Highest Average NEAT 40k',
                                      'Lowest Average NEAT 40k'])
    dfAverage['Evaluations'] = dfMainNEAT['Evaluations']
    dfAverage['Average NEAT 20k'] = dfMainNEAT['Average NEAT 20k'] / referenceFitness
    dfAverage['Average NEAT 40k'] = dfMainNEAT2['Average NEAT 40k'] / referenceFitness
    dfAverage['Highest Average NEAT 20k'] = dfMainNEATBestLineAverage['Highest Average NEAT 20k'] / referenceFitness
    dfAverage['Lowest Average NEAT 20k'] = dfMainNEATBestLineAverage['Lowest Average NEAT 20k'] / referenceFitness
    dfAverage['Highest Average NEAT 40k'] = dfMainNEATBestLineAverage2['Highest Average NEAT 40k'] / referenceFitness
    dfAverage['Lowest Average NEAT 40k'] = dfMainNEATBestLineAverage2['Lowest Average NEAT 40k'] / referenceFitness
    dfAverage['Average HyperNEAT 20k'] = dfMainHyperNEAT['Average HyperNEAT 20k'] / referenceFitness
    dfAverage['Average HyperNEAT 40k'] = dfMainHyperNEAT2['Average HyperNEAT 40k'] / referenceFitness
    dfAverage['Highest Average HyperNEAT 20k'] = dfMainHyperNEATBestLineAverage['Highest Average HyperNEAT 20k'] / referenceFitness
    dfAverage['Lowest Average HyperNEAT 20k'] = dfMainHyperNEATBestLineAverage['Lowest Average HyperNEAT 20k'] / referenceFitness
    dfAverage['Highest Average HyperNEAT 40k'] = dfMainHyperNEATBestLineAverage2['Highest Average HyperNEAT 40k'] / referenceFitness
    dfAverage['Lowest Average HyperNEAT 40k'] = dfMainHyperNEATBestLineAverage2['Lowest Average HyperNEAT 40k'] / referenceFitness

    ## Plot Average Graph
    plt.plot(dfAverage['Evaluations'], dfAverage['Average NEAT 20k'])
    plt.fill_between(dfAverage['Evaluations'], dfAverage['Lowest Average NEAT 20k'],
                     dfAverage['Highest Average NEAT 20k'], alpha=0.3)
    plt.plot(dfAverage['Evaluations'], dfAverage['Average NEAT 40k'])
    plt.fill_between(dfAverage['Evaluations'], dfAverage['Lowest Average NEAT 40k'],
                     dfAverage['Highest Average NEAT 40k'], alpha=0.3)
    plt.plot(dfAverage['Evaluations'], dfAverage['Average HyperNEAT 20k'])
    plt.fill_between(dfAverage['Evaluations'], dfAverage['Lowest Average HyperNEAT 20k'],
                     dfAverage['Highest Average HyperNEAT 20k'], alpha=0.3)
    plt.plot(dfAverage['Evaluations'], dfAverage['Average HyperNEAT 40k'])
    plt.fill_between(dfAverage['Evaluations'], dfAverage['Lowest Average HyperNEAT 40k'],
                     dfAverage['Highest Average HyperNEAT 40k'], alpha=0.3)
    plt.title('Mean performance')
    plt.ylabel('Average Fitness')
    plt.xlabel('Evaluations ($x10^6$)')
    plt.legend(['NEAT 20k', 'NEAT 40k', 'HyperNEAT 20k', 'HyperNEAT40k'])
    plt.savefig("Mean Performance.png")
    plt.show()

    dfCoverage = pd.DataFrame(columns=['Evaluations', 'Coverage NEAT 20k', 'Coverage NEAT 40k'])
    dfCoverage['Evaluations'] = dfMainNEAT['Evaluations']
    dfCoverage['Coverage NEAT 20k'] = dfMainNEAT['Niches'] / 200
    dfCoverage['Coverage NEAT 40k'] = dfMainNEAT2['Niches'] / 400
    dfCoverage['Coverage HyperNEAT 20k'] = dfMainHyperNEAT['Niches'] / 200
    dfCoverage['Coverage HyperNEAT 40k'] = dfMainHyperNEAT2['Niches'] / 400
    dfCoverage.plot.line(x="Evaluations")
    plt.title('Coverage')
    plt.ylabel('Coverage (%)')
    plt.xlabel('Evaluations ($x10^6$)')
    plt.legend(['NEAT 20k', 'NEAT 40k', 'HyperNEAT 20k', 'HyperNEAT40k'])
    plt.savefig('Coverage.png')
    plt.show()


def quality_metrics():
    plt.rcParams['figure.figsize'] = [4,5]
    plt.rcParams['figure.dpi'] = 100

    ## Load all 20k Map Fitness in
    dfReadIn = pd.read_csv(
        rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_20000archive\archive8011476.dat',
        header=None, sep='\s', engine='python')
    dfMainNEAT = pd.DataFrame(columns=['Fitness 0'])
    dfMainNEAT['Fitness 0'] = dfReadIn[dfReadIn.columns[0]]
    for i in range(1, 5):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\{i}_20000archive\archive8011476.dat'
        dfReadIn = pd.read_csv(filename, header=None, sep='\s', engine='python')
        dfMainNEAT.insert(i, 'Fitness ' + str(i), dfReadIn[dfReadIn.columns[0]], True)

    dfMainNEAT.to_csv('NEAT20kQualityMetrics.csv', index=False)

    ## Load all 40k Map Fitnesses in
    dfReadIn2 = pd.read_csv(
        rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_40000archive\archive8011476.dat',
        header=None, sep='\s', engine='python')
    dfMainNEAT2 = pd.DataFrame(columns=['Fitness 0'])
    dfMainNEAT2['Fitness 0'] = dfReadIn2[dfReadIn2.columns[0]]
    for i in range(1, 5):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\{i}_40000archive\archive8011476.dat'
        dfReadIn2 = pd.read_csv(filename, header=None, sep='\s', engine='python')
        dfMainNEAT2.insert(i, 'Fitness ' + str(i), dfReadIn2[dfReadIn.columns[0]], True)

    dfMainNEAT2.to_csv('NEAT40kQualityMetrics.csv', index=False)

    ## Load all 20k HyperNEAT Map Fitness in
    dfReadIn3 = pd.read_csv(
        rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\0_20000archive\archive8011438.dat',
        header=None, sep='\s', engine='python')
    dfMainHyperNEAT = pd.DataFrame(columns=['Fitness 0'])
    dfMainHyperNEAT['Fitness 0'] = dfReadIn3[dfReadIn3.columns[0]]
    for i in range(1, 5):
        if i == 1 or i == 4:
            filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\{i}_20000archive\archive8011438.dat'
        else:
            filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\{i}_20000archive\archive8001878.dat'
        dfReadIn3 = pd.read_csv(filename, header=None, sep='\s', engine='python')
        dfMainHyperNEAT.insert(i, 'Fitness ' + str(i), dfReadIn3[dfReadIn.columns[0]], True)

    dfMainHyperNEAT.to_csv('HyperNEAT20kQualityMetrics.csv', index=False)

    ## Load all 40k HyperNEAT Map Fitnesses in
    dfReadIn4 = pd.read_csv(
        rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\0_40000archive\archive8001878.dat',
        header=None, sep='\s', engine='python')
    dfMainHyperNEAT2 = pd.DataFrame(columns=['Fitness 0'])
    dfMainHyperNEAT2['Fitness 0'] = dfReadIn4[dfReadIn4.columns[0]]
    for i in range(1, 5):
        if i == 3:
            filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\{i}_40000archive\archive8011438.dat'
        else:
            filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\HyperNEAT\{i}_40000archive\archive8001878.dat'
        dfReadIn4 = pd.read_csv(filename, header=None, sep='\s', engine='python')
        dfMainHyperNEAT2.insert(i, 'Fitness ' + str(i), dfReadIn4[dfReadIn.columns[0]], True)

    dfMainHyperNEAT2.to_csv('HyperNEAT40kQualityMetrics.csv', index=False)

    ## Find reference Gait (Highest Performing Gait) Fitness
    referenceFitness = max(dfMainNEAT.max(axis=0).append(dfMainNEAT2.max(axis=0)))

    ## Find normalised best performing gaits per map per map size
    perfromanceNEAT = dfMainNEAT.max(axis=0) / referenceFitness
    perfromanceNEAT2 = dfMainNEAT2.max(axis=0) / referenceFitness
    perfromanceHyperNEAT = dfMainHyperNEAT.max(axis=0) / referenceFitness
    perfromanceHyperNEAT2 = dfMainHyperNEAT2.max(axis=0) / referenceFitness


    ## Set up Performance Plot
    boxes = [(plt.boxplot(perfromanceNEAT, positions=[0.3], patch_artist=True)),
             (plt.boxplot(perfromanceNEAT2, positions=[0.85], patch_artist=True)),
             (plt.boxplot(perfromanceHyperNEAT, positions=[1.45], patch_artist=True)),
             (plt.boxplot(perfromanceHyperNEAT2, positions=[2], patch_artist=True))]

    # Plot performance
    colours = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
    count = 0
    for box in boxes:
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color='black')
        plt.setp(box["boxes"], facecolor=colours[count])
        plt.setp(box["fliers"], markeredgecolor=colours[count])
        count += 1
    plt.xticks([0.3, 0.85, 1.45, 2], ['NEAT 20k', 'NEAT 40k', 'HNEAT 20k', 'HNEAT 40k'], fontsize=8)
    plt.ylim([0, 1.02]) # Option to scale but it looks dumb
    plt.title("Performance")
    plt.savefig("Performance.png")
    plt.show()

    ## Set up for reliability and precision
    column_list = ['Fitness 0', 'Fitness 1', 'Fitness 2', 'Fitness 3', 'Fitness 4']
    precisionArrayNEAT = np.array(dfMainNEAT[column_list].mean(axis=0) / referenceFitness)
    precisionArrayNEAT2 = np.array(dfMainNEAT2[column_list].mean(axis=0) / referenceFitness)
    precisionArrayHyperNEAT = np.array(dfMainHyperNEAT[column_list].mean(axis=0) / referenceFitness)
    precisionArrayHyperNEAT2 = np.array(dfMainHyperNEAT2[column_list].mean(axis=0) / referenceFitness)
    reliabilityArrayNEAT = np.array((dfMainNEAT[column_list].sum(axis=0) / 20000) / referenceFitness)
    reliabilityArrayNEAT2 = np.array((dfMainNEAT2[column_list].sum(axis=0) / 40000) / referenceFitness)
    reliabilityArrayHyperNEAT = np.array((dfMainHyperNEAT[column_list].sum(axis=0) / 20000) / referenceFitness)
    reliabilityArrayHyperNEAT2 = np.array((dfMainHyperNEAT2[column_list].sum(axis=0) / 40000) / referenceFitness)
    boxPlotSetUpReliability = [reliabilityArrayNEAT, reliabilityArrayNEAT2, reliabilityArrayHyperNEAT, reliabilityArrayHyperNEAT2]
    boxPlotSetUpPrecision = [precisionArrayNEAT, precisionArrayNEAT2, precisionArrayHyperNEAT, precisionArrayHyperNEAT2]

    # Plot Reliability
    colours = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
    boxes = [(plt.boxplot(boxPlotSetUpReliability[0], positions=[0.3], patch_artist=True)),
             (plt.boxplot(boxPlotSetUpReliability[1], positions=[0.85], patch_artist=True)),
             (plt.boxplot(boxPlotSetUpReliability[2], positions=[1.45], patch_artist=True)),
             (plt.boxplot(boxPlotSetUpReliability[3], positions=[2], patch_artist=True))
             ]
    count = 0
    for box in boxes:
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color='black')
        plt.setp(box["boxes"], facecolor=colours[count])
        plt.setp(box["fliers"], markeredgecolor=colours[count])
        count += 1
    plt.xticks([0.3, 0.85, 1.45, 2], ['NEAT 20k', 'NEAT 40k', 'HNEAT 20k', 'HNEAT 40k'], fontsize=8)
    plt.ylim([0, 1.02]) # Option to scale but it looks dumb
    plt.title("Reliability")
    plt.savefig("Reliability.png")
    plt.show()

    ## Plot precision
    colours = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
    boxes = [(plt.boxplot(boxPlotSetUpPrecision[0], positions=[0.3], patch_artist=True)),
             (plt.boxplot(boxPlotSetUpPrecision[1], positions=[0.85], patch_artist=True)),
             (plt.boxplot(boxPlotSetUpPrecision[2], positions=[1.45], patch_artist=True)),
             (plt.boxplot(boxPlotSetUpPrecision[3], positions=[2], patch_artist=True))
             ]
    count = 0
    for box in boxes:
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color='black')
        plt.setp(box["boxes"], facecolor=colours[count])
        plt.setp(box["fliers"], markeredgecolor=colours[count])
        count += 1
    plt.xticks([0.3, 0.85, 1.45, 2], ['NEAT 20k', 'NEAT 40k', 'HNEAT 20k', 'HNEAT 40k'], fontsize=8)
    plt.ylim([0, 1.02]) # Option to scale but it looks dumb
    plt.title("Precision")
    plt.savefig("Precision.png")
    plt.show()

    ## Setup coverage
    coverageNEAT = np.array([dfMainNEAT[dfMainNEAT.columns[0]].count()])
    coverageNEAT2 = np.array([dfMainNEAT2[dfMainNEAT2.columns[0]].count()])
    coverageHyperNEAT = np.array([dfMainHyperNEAT[dfMainHyperNEAT.columns[0]].count()])
    coverageHyperNEAT2 = np.array([dfMainHyperNEAT2[dfMainHyperNEAT2.columns[0]].count()])
    for i in range(1, 5):
        coverageNEAT = np.append(coverageNEAT, dfMainNEAT[dfMainNEAT.columns[i]].count())
        coverageNEAT2 = np.append(coverageNEAT2, dfMainNEAT2[dfMainNEAT2.columns[i]].count())
        coverageHyperNEAT = np.append(coverageHyperNEAT, dfMainHyperNEAT[dfMainHyperNEAT.columns[i]].count())
        coverageHyperNEAT2 = np.append(coverageHyperNEAT2, dfMainHyperNEAT2[dfMainHyperNEAT2.columns[i]].count())
    coverageNEAT = coverageNEAT / 20000
    coverageNEAT2 = coverageNEAT2 / 40000
    coverageHyperNEAT = coverageHyperNEAT / 20000
    coverageHyperNEAT2 = coverageHyperNEAT2 / 40000
    print(np.mean(coverageNEAT)*100)
    print(np.mean(coverageNEAT2)*100)
    print(np.mean(coverageHyperNEAT)*100)
    print(np.mean(coverageHyperNEAT2)*100)

    ## Plot coverage
    colours = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
    boxes = [(plt.boxplot(coverageNEAT, positions=[0.3], patch_artist=True)),
             (plt.boxplot(coverageNEAT2, positions=[0.85], patch_artist=True)),
             (plt.boxplot(coverageHyperNEAT, positions=[1.45], patch_artist=True)),
             (plt.boxplot(coverageHyperNEAT2, positions=[2], patch_artist=True))
             ]
    count = 0
    for box in boxes:
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color='black')
        plt.setp(box["boxes"], facecolor=colours[count])
        plt.setp(box["fliers"], markeredgecolor=colours[count])
        count += 1
    plt.xticks([0.3, 0.85, 1.45, 2], ['NEAT 20k', 'NEAT 40k', 'HNEAT 20k', 'HNEAT 40k'], fontsize=8)
    plt.ylim([0, 1.02])  # Option to scale but it looks dumb
    plt.title("Coverage")
    plt.savefig("Coverage Map Metric.png")
    plt.show()


progression_metrics()
#quality_metrics()
