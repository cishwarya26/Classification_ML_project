import numpy as np
import pandas as pd
import os
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.preprocessing import robust_scale


project_dir = '/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/'
input_file = '/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/data.csv'


dataset=pd.read_csv(input_file, sep=',',names = ["fLength", "fWidth", "fSize", "fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"])

# create design matrix X and target vector y
X = np.array(dataset.ix[:, 0:10])   # end index is exclusive
y = np.array(dataset['class'])  # another way of indexing a pandas df

scaled_x=robust_scale(X, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)

#splitting
x_train, x_test, y_train, y_test = train_test_split(scaled_x, y, test_size=0.33, random_state=42)

#svm function

def svm_algo(gamma,c):   
    clf = svm.SVC(gamma=gamma,C=c,probability=True)
    y_train2 = np.ravel(y_train)
    clf = clf.fit(x_train, y_train2)
    y_pred = clf.predict(x_test)
    accur=accuracy_score(y_test, y_pred)
    return [accur,y_pred,clf]

def calcuate_confusionMatrix(case, result):
    print ('Case: ', case)
    # Confusion matrix
    confusionMatrix = confusion_matrix(y_test, result, labels=['g','h'])
    print('Confusion Matrix: \n', confusionMatrix)
    
    plt.matshow(confusionMatrix)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(project_dir+case+'_confusionMatrix.png')
    
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

def evaluationMeasures(classifier_type):
	print("===============================================================================")
	print("\nEvaluation Measures for",classifier_type,":")
	print("===============================================================================")
	print("Classification Report:\n\n",classification_report(y_test, y_pred2))

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

def roc_and_auc(clacf2):
	#clacf2=result_1[2]
	prob2=clacf2.predict_proba(x_test)[:,1]

	#transforming the target data into binary 
	lb = preprocessing.LabelBinarizer()
	y_test_bin=lb.fit_transform(y_test)

	#plot the ROC curve
	fpr, tpr, thresholds= metrics.roc_curve(y_test_bin,prob2)
	plt.plot(fpr, tpr)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for the SVM classifier')
	plt.grid(True)
	#plt.show()
	plt.savefig(project_dir+'roc_curve_svm.png')
	print("===============================================================================")
	#AUC score for SVM
	print ('This is the AUC for the SVM machine')
	print (metrics.roc_auc_score(y_test_bin,prob2))

#Running Cases for SVM

result_3=svm_algo(0.001,100)
result_4=svm_algo(0.0001,100)
print("===============================================================================")
print ('Comparing Accuracy for SVM:')
print("===============================================================================")

max_res=max(result_3[0],result_4[0])

if result_3[0] > result_4[0]:
        y_pred2=result_3[1]
        print ('The best parameters values: C=0.001,gamma=1000 and accuracy:',result_3[0])
        clacf2=result_3[2]
        roc_and_auc(clacf2)
else:
        y_pred2=result_4[1]
        print ('The best parameters values: C=0.0001,gamma=1000 and accuracy:',result_4[0])
        clacf2=result_4[2]
        roc_and_auc(clacf2)

#calculating the probabilities

confusionMatrix =calcuate_confusionMatrix('SVM', y_pred2)
evaluationMeasures("SVM")














