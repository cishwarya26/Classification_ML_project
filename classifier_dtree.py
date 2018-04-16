# Data Mining Project 1

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
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import robust_scale
from sklearn.metrics import classification_report
#input_file = '/home/wajrcs/Downloads/magic04.data'
#project_dir = '/var/www/data-mining/'

project_dir = '/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/'
input_file = '/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/data.csv'

# Comma delimited is the default
#df = pd.read_csv(input_file, header=0)
#df_target = pd.read_csv(input_file, header=0, usecols=[10])

# Remove the non-numeric columns
#df_data = df._get_numeric_data()

dataset=pd.read_csv(input_file, sep=',',names = ["fLength", "fWidth", "fSize", "fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"])
X = np.array(dataset.ix[:, 0:10])   # end index is exclusive
y = np.array(dataset['class'])  # another way of indexing a pandas df

scaled_x=robust_scale(X, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)

#do the slpitting
x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.33, random_state=42)

# Method to calculate confusionMatrix
def calcuate_confusionMatrix(case, result):
	
	# Confusion matrix
	confusionMatrix = confusion_matrix(y_test, result, labels=['g','h'])
	print('Confusion Matrix for',case,':\n', confusionMatrix)

	plt.matshow(confusionMatrix)
	plt.title('Confusion Matrix')
	plt.colorbar()
	plt.ylabel('True class')
	plt.xlabel('Predicted class')
	plt.savefig(project_dir+'classifier/'+case+'_confusionMatrix.png')

	#[row, column]
	TP = confusionMatrix[1, 1]
	print ("TP: ",TP)
	TN = confusionMatrix[0, 0]
	print ("TN: ",TN)
	FP = confusionMatrix[0, 1]
	print ("FP: ",FP)
	FN = confusionMatrix[1, 0]
	print ("FN: ",FN)

	return [TP, TN, FP, FN]

# Method to apply Decssion Tree Classifier
def decision_tree(param):
	# Applying Classifier
	clf = None

	if (param == 'gini'):
		print ("Case Gini:")
		clf = tree.DecisionTreeClassifier(criterion='gini', random_state = 100, max_depth=7, min_samples_leaf=5) 

	if (param == 'entropy'):
		print ("Case Entropy:")
		clf = tree.DecisionTreeClassifier(criterion='entropy', random_state = 100, max_depth=7, min_samples_leaf=5)

	# Doing prediction
	clf = clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	print ('Result Predicted: ', y_pred)

	with open(project_dir+'classifier/dtree'+param+'.dot', 'w') as f:
		f = tree.export_graphviz(clf, out_file=f)

	# Use this command to generate plot
	# dot -Tpdf iris.dot -o iris.pdf

	# Calculating accuracy
	cal_accuracy_score = accuracy_score(y_test, y_pred)
	print("Accuracy Score: \n", cal_accuracy_score)

	return [cal_accuracy_score, y_pred]

def evaluationMeasures(classifier_type):
	print("\nEvaluation Measures for",classifier_type,":")
	print("===============================================================================")
	print("Classification Report:\n\n",classification_report(y_test, y_pred_1))

	classification_error = (confusionMatrix[2]+confusionMatrix[3]) / float(confusionMatrix[0]+confusionMatrix[1]+confusionMatrix[2]+confusionMatrix[2])
	print("Classification Error:", classification_error)

	specificity = confusionMatrix[1] / (confusionMatrix[1]+confusionMatrix[2])
	print("Specificity", specificity)

	sensitivity = confusionMatrix[0] / float(confusionMatrix[0] + confusionMatrix[3])
	recall = sensitivity
	print("Sensitivity:",sensitivity)
	print("Recall:",recall)

	precision = confusionMatrix[0] / float(confusionMatrix[0] + confusionMatrix[2])
	print("Precision:",precision)

	F1 = 2 * (precision * recall) / (precision + recall)
	print("F1 score:",F1)

def roc_and_auc(criteria):
	clf = tree.DecisionTreeClassifier(criterion=criteria, random_state = 100, max_depth=7, min_samples_leaf=5) 
	clf = clf.fit(x_train, y_train)
	prob2=clf.predict_proba(x_test)[:,1]

	#transforming the target data into binary 
	lb = preprocessing.LabelBinarizer()
	y_test_bin=lb.fit_transform(y_test)

	print("===============================================================================")
	#plot the ROC curve
	fpr, tpr, thresholds= metrics.roc_curve(y_test_bin,prob2)
	plt.plot(fpr, tpr)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for the Dtree classifier')
	plt.grid(True)
	#plt.show()
	plt.savefig(project_dir+'roc_curve_dtree.png')
	print("===============================================================================")
	#AUC score for SVM
	print ('This is the AUC for the Decision tree classifier')
	print (metrics.roc_auc_score(y_test_bin,prob2))

# Running Cases for decision tree
results_gini = decision_tree('gini');
results_entropy = decision_tree('entropy');
# Comparing accuracy
y_pred_1 = None
print("===============================================================================")
print ('Comparing Accuracy for Decision tree:')
print("===============================================================================")
if (results_gini[0] > results_entropy[0]):
	y_pred_1 = results_gini[1]
	print ('Case Gini is better.')
	roc_and_auc('gini')
else:
	y_pred_1 = results_entropy[1]
	print ('Case Entropy is better.')
	roc_and_auc('entropy')
     
#calculating confusion matrix    
confusionMatrix=calcuate_confusionMatrix('DecisionTree', y_pred_1)
evaluationMeasures("DecisionTree")


















