# Functions specific to UCI dataset 'MAGIC Gamma Telescope Data Set'
# For COMP551 mini-project 1
# January 30, 2020
# Lia Formenti

# It would be cool to make a data class, with attribues feature_names, file_name, a method for loading and cleaning . . . 

import pandas as pd

def returnFeatureNamesList():
    return ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1',        'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

def returnFileName():
    return 'magic04.data'

def loadMagic():
    featureNames = returnFeatureNamesList()
    fileName = returnFileName()
    translate = {'g':1, 'h':0} # assign integer classes to orig. classes
    frame = pd.read_csv(fileName, header=0, names=featureNames)
    # print(frame)
    # Randomize the rows to randomize class order
    # frame = frame.sample(frac=1).reset_index(drop=True)
    # Set signal vs background class to bools, true for signal (gamma, 'g')
    frame['class'] = frame['class'].map(translate)
    return frame

