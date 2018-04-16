# loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.cross_validation import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import recall_score
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import roc_auc_score
#import collections
from pandas.tools.plotting import scatter_matrix
#from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.metrics import classification_report
dataset = pd.read_csv("/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/data.csv", sep=',', 
                  names = ["fLength", "fWidth", "fSize", "fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"])

project_dir = '/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/'

# shape
instance_count, attr_count = dataset.shape
dataset.describe()
dataset.count()
dataset.min()
dataset.max()
dataset.median()
#dataset.quantile(q)

print("Any missing value",pd.isnull(dataset).any())
print("Dataset shape:",dataset.shape)

# class distribution
print("Class distribution:",dataset.groupby('class').size())

# scatter plot matrix
#scatter_matrix(dataset)
#plt.savefig(project_dir+'scatterPlot.png')

#Pearson coefficient
pearson=dataset.corr(method='pearson')
print("Pearson:",pearson)
# assume target attr is the last, then remove corr with itself
corr_with_target = pearson.ix[-1][:-1]
#attributes sorted from the most predictive
predictivity = corr_with_target.sort_values(ascending=False)

#sort the correlations by the absolute value
corr_with_target[abs(corr_with_target).argsort()[::-1]]

#dataset.groupby('class').hist()
#plt.savefig(project_dir+'classGrouping.png')
dataset.groupby('class').fLength.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fLength.png')
dataset.groupby('class').fWidth.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fWidth.png')
dataset.groupby('class').fSize.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fSize.png')
dataset.groupby('class').fConc.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fConc.png')
dataset.groupby('class').fConc1.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fConc1.png')
dataset.groupby('class').fAsym.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fAsym.png')
dataset.groupby('class').fM3Long.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fM3Long.png')
dataset.groupby('class').fM3Trans.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fM3Trans.png')
dataset.groupby('class').fAlpha.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fAlpha.png')
dataset.groupby('class').fDist.hist(alpha=0.4)
plt.show()
#plt.savefig(project_dir+'fDist.png')


