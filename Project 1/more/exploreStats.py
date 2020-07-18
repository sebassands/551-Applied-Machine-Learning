# Functions for exploring the statistics of a dataset
# Dataset is a pandas dataframe
# Janurary 30, 2020
# Lia Formenti

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from magicGammaFunctions import loadMagic

# Randomized, bool classes
magicFrame = loadMagic()

# print(magicFrame.describe())
# Get description
# f = open('describeMagic.txt', 'w')
# magicFrame.describe().to_csv(f, sep=' ', float_format="%.2f")
# f.close()

# small scale boxplots
# magicFrame.boxplot(column=['fSize', 'fConc', 'fConc1'])

# large scale boxplots
# magicFrame.boxplot(column=['fLength', 'fWidth', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist'])

# All data histograms for all features
# magicFrame.hist()

# Now plot by class group. Explicitly get each class for clarity.
# This was when classes were denoted by chars
# grouped = magicFrame.groupby('class')
# grouped.get_group('g').hist()

# Histograms by (bool) classes
# magicFrame.groupby('class').hist()

# Overlaid histograms by class
# '''
figure, axes = plt.subplots(1,1)
magicFrame.groupby('class')['fAlpha'].hist(alpha=0.4)
#
#figure, axes = plt.subplots(nrows=4, ncols=3)
#for colNum, colName in enumerate(magicNames[:-1], start=0):
#    for name, group in grouped:
#        group[colName].hist(alpha=0.4, ax=axes[colNum%4, colNum//4], label=name)
#    axes[colNum%4, colNum//4].set_title(colName)
#    axes[colNum%4, colNum//4].legend()
#plt.show()
## '''
# Scatter matrix
# pd.plotting.scatter_matrix(magicFrame.iloc[:,:-1],c=['r' if i == True else 'b' for i in magicFrame['class']], alpha=0.2, figsize=(11,11))

# Individual scatter matrices of interest
# magicFrame.plot.scatter(x='fLength', y='fSize', c = [ 'r' if b == True else 'b' for b in magicFrame['class']], alpha=0.1)
# magicFrame.plot.scatter(x='fLength', y='fAlpha', c = [ 'r' if b == True else 'b' for b in magicFrame['class']], alpha=0.1)
# magicFrame.plot.scatter(x='fWidth', y='fSize', c = [ 'r' if b == True else 'b' for b in magicFrame['class']], alpha=0.1)
# magicFrame.plot.scatter(x='fWidth', y='fConc', c = [ 'r' if b == True else 'b' for b in magicFrame['class']], alpha=0.1)
# magicFrame.plot.scatter(x='fWidth', y='fAlpha', c = [ 'r' if b == True else 'b' for b in magicFrame['class']], alpha=0.1)
# magicFrame.plot.scatter(x='fSize', y='fM3Trans', c = [ 'r' if b == True else 'b' for b in magicFrame['class']], alpha=0.1)
# magicFrame.plot.scatter(x='fAsym', y='fM3Trans', c = [ 'r' if b == True else 'b' for b in magicFrame['class']], alpha=0.1)

plt.show()
