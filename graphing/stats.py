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

dfMainBest = pd.DataFrame(columns=['Best NEAT 0', 'Best NEAT 1', 'Best NEAT 2', 'Best NEAT 3', 'Best NEAT 4',
                               'Best NEAT 5', 'Best NEAT 6', 'Best NEAT 7', 'Best NEAT 8', 'Best NEAT 9',
                               'Best HyperNEAT 0', 'Best HyperNEAT 1', 'Best HyperNEAT 2', 'Best HyperNEAT 3',
                               'Best HyperNEAT 4',
                               'Best HyperNEAT 5', 'Best HyperNEAT 6', 'Best HyperNEAT 7', 'Best HyperNEAT 8',
                               'Best HyperNEAT 9',
                               ])

for i in range(0, 20):
    NEATfilename = r'C:\Users\micha\PycharmProjects\Honours Project\NEATOutput\genomeFitness\NEATFitnessHistory' + str(
        i) + '.csv'
    HyperNEATfilename = r'C:\Users\micha\PycharmProjects\Honours Project\HyperNEATOutput\genomeFitness\HyperNEATFitnessHistory' + str(i) + '.csv'
    dfNEAT = pd.read_csv(NEATfilename)
    dfHyperNEAT = pd.read_csv(HyperNEATfilename)
    dfNEAT.columns = ['Best NEAT', 'Average NEAT']
    dfHyperNEAT.columns = ['Best HyperNEAT', 'Average HyperNEAT']
    dfMainBest['Best NEAT ' + str(i)] = dfNEAT['Best NEAT']
    dfMainBest['Best HyperNEAT ' + str(i)] = dfHyperNEAT['Best HyperNEAT']
    #dfMain['Average NEAT'] = dfMain['Average NEAT'] + df['Average NEAT']
dfMainBest.reset_index()

for i in range(1999):
    print(i)
    NEATvals = dfMainBest.iloc[i, 0:10]
    HyperNEATvals = dfMainBest.iloc[i, 10:20]
    print(sp.stats.ranksums(NEATvals, HyperNEATvals, 'greater'))




