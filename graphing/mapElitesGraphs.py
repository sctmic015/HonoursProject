#| Author: Michael Scott

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from io import StringIO
import webbrowser
from tempfile import NamedTemporaryFile




def progression_metrics():
    dfMain = pd.read_csv(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_20000\log.dat',
            header=None, sep='\s', engine='python')
    dfMain.columns = ['Evaluations', 'Niches', 'Best NEAT 20k', 'Average NEAT 20k', 'Median NEAT 20k', '5th percentile', '95th percentile']
    dfMain = dfMain.drop(dfMain.index[3349:len(dfMain)])
    dfMainBestLine = pd.DataFrame(columns=['Best NEAT 20k'])
    dfMainBestLine['Best NEAT 20k'] = dfMain['Best NEAT 20k']
    dfMainBestLineAverage = pd.DataFrame(columns=['Average NEAT 20k'])
    dfMainBestLineAverage['Average NEAT 20k'] = dfMain['Average NEAT 20k']
    for i in range(1, 5):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\{i}_20000\log.dat'
        df = pd.read_csv(filename, header=None, sep='\s', engine='python')
        df.columns = ['Evaluations', 'Niches', 'Best NEAT 20k', 'Average NEAT 20k', 'Median NEAT 20k', '5th percentile', '95th percentile']
        dfMain = dfMain + df
        dfMainBestLine.insert(i, 'Best NEAT 20k ' + str(i), df['Best NEAT 20k'], True)
        dfMainBestLineAverage.insert(i, 'Average NEAT 20k' + str(i), df['Average NEAT 20k'], True)
        #dfMainBestLine['Best NEAT 20k ' + str(i)] = df['Best NEAT 20k']
    dfMainBestLine.insert(5, 'Highest NEAT 20k', dfMainBestLine.max(axis = 1), True)
    dfMainBestLine.insert(5, 'Lowest NEAT 20k', dfMainBestLine.min(axis = 1), True)
    dfMainBestLineAverage.insert(5, 'Highest Average NEAT 20k', dfMainBestLineAverage.max(axis=1), True)
    dfMainBestLineAverage.insert(5, 'Lowest Average NEAT 20k', dfMainBestLineAverage.min(axis=1), True)
    dfMain = dfMain / 5



    dfMain.reset_index()
    # print(dfMainBestLine)
    # print(dfMainBestLineAverage)


    dfMain2 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_40000\log.dat',
            header=None, sep='\s', engine='python')
    dfMain2.columns = ['Evaluations', 'Niches', 'Best NEAT 40k', 'Average NEAT 40k', 'Median NEAT 40k', '5th percentile', '95th percentile']
    dfMain2 = dfMain2.drop(dfMain2.index[3349:len(dfMain)])
    dfMainBestLine2 = pd.DataFrame(columns=['Best NEAT 40k'])
    dfMainBestLine2['Best NEAT 40k'] = dfMain2['Best NEAT 40k']
    dfMainBestLineAverage2 = pd.DataFrame(columns=['Average NEAT 40k'])
    dfMainBestLineAverage2['Average NEAT 40k'] = dfMain2['Average NEAT 40k']
    for i in range(1, 5):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\{i}_40000\log.dat'
        df = pd.read_csv(filename, header=None, sep='\s', engine='python')
        df.columns = ['Evaluations', 'Niches', 'Best NEAT 40k', 'Average NEAT 40k', 'Median NEAT 40k', '5th percentile', '95th percentile']
        dfMain2 = dfMain2 + df
        dfMainBestLine2.insert(i, 'Best NEAT 40k ' + str(i), df['Best NEAT 40k'], True)
        dfMainBestLineAverage2.insert(i, 'Average NEAT 40k' + str(i), df['Average NEAT 40k'], True)
    dfMainBestLine2.insert(5, 'Highest NEAT 40k', dfMainBestLine2.max(axis=1), True)
    dfMainBestLine2.insert(5, 'Lowest NEAT 40k', dfMainBestLine2.min(axis=1), True)
    dfMainBestLineAverage2.insert(5, 'Highest Average NEAT 40k', dfMainBestLineAverage2.max(axis=1), True)
    dfMainBestLineAverage2.insert(5, 'Lowest Average NEAT 40k', dfMainBestLineAverage2.min(axis=1), True)
    dfMain2 = dfMain2 / 5
    dfMain2.reset_index()

    dfBest = pd.DataFrame(columns=['Evaluations', 'Best NEAT 20k', 'Best NEAT 40k', 'Highest NEAT 40k', 'Lowest NEAT 40k'])
    dfBest['Evaluations'] = dfMain['Evaluations']
    dfBest['Best NEAT 20k'] = dfMain['Best NEAT 20k']
    dfBest['Best NEAT 40k'] = dfMain2['Best NEAT 40k']
    dfBest['Highest NEAT 20k'] = dfMainBestLine['Highest NEAT 20k']
    dfBest['Lowest NEAT 20k'] = dfMainBestLine['Lowest NEAT 20k']
    dfBest['Highest NEAT 40k'] = dfMainBestLine2['Highest NEAT 40k']
    dfBest['Lowest NEAT 40k'] = dfMainBestLine2['Lowest NEAT 40k']

    #dfBest.plot.line(x='Evaluations')
    plt.plot(dfBest['Evaluations'], dfBest['Best NEAT 20k'])
    plt.fill_between(dfBest['Evaluations'], dfBest['Lowest NEAT 20k'], dfBest['Highest NEAT 20k']+0.01, alpha = 0.2)
    plt.plot(dfBest['Evaluations'], dfBest['Best NEAT 40k'])
    plt.fill_between(dfBest['Evaluations'], dfBest['Lowest NEAT 40k'], dfBest['Highest NEAT 40k'] + 0.01, alpha=0.2)
    plt.ylim(7, 8)
    plt.title('Best Performance')
    plt.ylabel('Best Fitness (m)')
    plt.xlabel('Evaluations ($x10^6$)')
    plt.show()

    dfAverage = pd.DataFrame(columns=['Evaluations', 'Average NEAT 20k', 'Average NEAT 40k', 'Highest Average NEAT 40k', 'Lowest Average NEAT 40k'])
    dfAverage['Evaluations'] = dfMain['Evaluations']
    dfAverage['Average NEAT 20k'] = dfMain['Average NEAT 20k']
    dfAverage['Average NEAT 40k'] = dfMain2['Average NEAT 40k']
    dfAverage['Highest Average NEAT 20k'] = dfMainBestLineAverage['Highest Average NEAT 20k']
    dfAverage['Lowest Average NEAT 20k'] = dfMainBestLineAverage['Lowest Average NEAT 20k']
    dfAverage['Highest Average NEAT 40k'] = dfMainBestLineAverage2['Highest Average NEAT 40k']
    dfAverage['Lowest Average NEAT 40k'] = dfMainBestLineAverage2['Lowest Average NEAT 40k']

    plt.plot(dfAverage['Evaluations'], dfAverage['Average NEAT 20k'])
    plt.fill_between(dfAverage['Evaluations'], dfAverage['Lowest Average NEAT 20k'], dfAverage['Highest Average NEAT 20k'], alpha = 0.2)
    plt.plot(dfAverage['Evaluations'],dfAverage['Average NEAT 40k'])
    plt.fill_between(dfAverage['Evaluations'], dfAverage['Lowest Average NEAT 40k'], dfAverage['Highest Average NEAT 40k'], alpha = 0.2)
    plt.title('Mean performance')
    plt.ylabel('Average Fitness (m)')
    plt.xlabel('Evaluations ($x10^6$)')
    plt.show()

    dfCoverage = pd.DataFrame(columns=['Evaluations', 'Coverage NEAT 20k', 'Coverage NEAT 40k'])
    dfCoverage['Evaluations'] = dfMain['Evaluations']
    dfCoverage['Coverage NEAT 20k'] = dfMain['Niches'] / 200
    dfCoverage['Coverage NEAT 40k'] = dfMain2['Niches'] / 400
    dfCoverage.plot.line(x="Evaluations")
    plt.title('Coverage')
    plt.ylabel('Coverage (%)')
    plt.xlabel('Evaluations ($x10^6$)')
    plt.show()

def quality_metrics():

    ## Load all 20k Map Fitness in
    dfReadIn = pd.read_csv(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_20000archive\archive8011476.dat',
                         header=None, sep='\s', engine='python')
    dfMain = pd.DataFrame(columns=['Fitness 0'])
    dfMain['Fitness 0'] = dfReadIn[dfReadIn.columns[0]]
    for i in range(1, 5):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\{i}_20000archive\archive8011476.dat'
        dfReadIn = pd.read_csv(filename, header=None, sep='\s', engine='python')
        dfMain.insert(i, 'Fitness ' + str(i), dfReadIn[dfReadIn.columns[0]], True)

    ## Load all 40k Map Fitnesses in
    dfReadIn2 = pd.read_csv(rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\0_40000archive\archive8011476.dat',
                         header=None, sep='\s', engine='python')
    dfMain2 = pd.DataFrame(columns=['Fitness 0'])
    dfMain2['Fitness 0'] = dfReadIn2[dfReadIn2.columns[0]]
    for i in range(1, 5):
        filename = rf'C:\Users\micha\PycharmProjects\Honours Project\mapElitesOutput\NEAT\{i}_40000archive\archive8011476.dat'
        dfReadIn2 = pd.read_csv(filename, header=None, sep='\s', engine='python')
        dfMain2.insert(i, 'Fitness ' + str(i), dfReadIn2[dfReadIn.columns[0]], True)


    ## Find reference Gait (Highest Performing Gait) Fitness
    referenceFitness = max(dfMain.max(axis=0).append(dfMain2.max(axis=0)))

    ## Find normalised best performing gaits per map per map size
    perfromance = dfMain.max(axis=0) / referenceFitness
    perfromance2 = dfMain2.max(axis=0) / referenceFitness

    ## Set up Performance Plot
    boxes = [(plt.boxplot(perfromance, positions=[0.5], patch_artist=True)),
     (plt.boxplot(perfromance2, positions=[1], patch_artist=True))]

    # Plot performance
    colours = [(1, 0, 0, 0.3), (0, 1, 0, 0.3)]
    count = 0
    for box in boxes:
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color='black')
        plt.setp(box["boxes"], facecolor=colours[count])
        plt.setp(box["fliers"], markeredgecolor=colours[count])
        count += 1
    plt.xticks([0.5, 1], ['NEAT 20k', 'NEAT 40k'])
    # plt.ylim([0, 1]) # Option to scale but it looks dumb
    plt.title("Performance")
    plt.show()



    ## Set up for reliability and precision
    column_list = ['Fitness 0', 'Fitness 1', 'Fitness 2', 'Fitness 3', 'Fitness 4']
    precisionArray = np.array(dfMain[column_list].mean(axis = 0)/referenceFitness)
    precisionArray2 = np.array(dfMain2[column_list].mean(axis = 0)/referenceFitness)
    reliabilityArray = np.array((dfMain[column_list].sum(axis = 0) /20000)/referenceFitness)
    reliabilityArray2 = np.array((dfMain2[column_list].sum(axis=0) /40000)/referenceFitness)
    boxPlotSetUpReliability= [reliabilityArray, reliabilityArray2]
    boxPlotSetUpPrecision= [precisionArray, precisionArray2]

    # Plot Reliability
    colours = [(1, 0, 0, 0.3), (0, 1, 0, 0.3)]
    boxes = [(plt.boxplot(boxPlotSetUpReliability[0], positions= [0.5],patch_artist=True)),(plt.boxplot(boxPlotSetUpReliability[1], positions = [1], patch_artist=True))]
    count = 0
    for box in boxes:
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color='black')
        plt.setp(box["boxes"], facecolor=colours[count])
        plt.setp(box["fliers"], markeredgecolor=colours[count])
        count +=1
    plt.xticks([0.5, 1], ['NEAT 20k', 'NEAT 40k'])
    # plt.ylim([0, 1]) # Option to scale but it looks dumb
    plt.title("Reliability")
    plt.show()

    ## Plot precision
    colours = [(1, 0, 0, 0.3), (0, 1, 0, 0.3)]
    boxes = [(plt.boxplot(boxPlotSetUpPrecision[0], positions=[0.5], patch_artist=True)),
             (plt.boxplot(boxPlotSetUpPrecision[1], positions=[1], patch_artist=True))]
    count = 0
    for box in boxes:
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color='black')
        plt.setp(box["boxes"], facecolor=colours[count])
        plt.setp(box["fliers"], markeredgecolor=colours[count])
        count += 1
    plt.xticks([0.5, 1], ['NEAT 20k', 'NEAT 40k'])
    # plt.ylim([0, 1]) # Option to scale but it looks dumb
    plt.title("Precision")
    plt.show()

    ## Setup coverage
    coverage = np.array([dfMain[dfMain.columns[0]].count()])
    coverage2 = np.array([dfMain2[dfMain2.columns[0]].count()])
    for i in range(1, 5):
        coverage = np.append(coverage, dfMain[dfMain.columns[i]].count())
        coverage2 = np.append(coverage2, dfMain2[dfMain2.columns[i]].count())
    coverage = coverage/20000
    coverage2 = coverage2/40000
    print(coverage)

    ## Plot coverage
    colours = [(1, 0, 0, 0.3), (0, 1, 0, 0.3)]
    boxes = [(plt.boxplot(coverage, positions=[0.5], patch_artist=True)),
             (plt.boxplot(coverage2, positions=[1], patch_artist=True))]
    count = 0
    for box in boxes:
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box[item], color='black')
        plt.setp(box["boxes"], facecolor=colours[count])
        plt.setp(box["fliers"], markeredgecolor=colours[count])
        count += 1
    plt.xticks([0.5, 1], ['NEAT 20k', 'NEAT 40k'])
    plt.ylim([0, 1.1]) # Option to scale but it looks dumb
    plt.title("Coverage")
    plt.show()


progression_metrics()
quality_metrics()


