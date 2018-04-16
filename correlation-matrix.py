# Python program for Correlation Matrix
# Data Mining Project 1
# 28th Jun, 2017
# Authors: Waqar Alamgir <wajrcs@gmail.com>, Laridi Sofiane, Ishwarya

# Importing packages
import numpy as np
import pandas as pd
import os
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Directories settings
project_dir = '/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/'
input_file = '/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/data.csv'

# Defining global variables here
labels = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym:', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']

# Comma delimited is the default
df = pd.read_csv(input_file, header=0)
df_target = pd.read_csv(input_file, header=0, usecols=[10])

# Remove the non-numeric columns
df_data = df._get_numeric_data()

# Plots correlation matrix
def plot_correlation_matrix(cm, title='Correlation Matrix', labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(df_data.corr())
    plt.title(title)
    fig.colorbar(cax)
    if labels:
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)

    plt.savefig(project_dir+'correlation-matrix/confusion_matrix.png')
    plt.show()

# Plot the Correlation Matrix
plot_correlation_matrix(df_data, 'Correlation Matrix', labels)