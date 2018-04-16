import numpy as np
import pandas as pd
import os
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn import model_selection
import collections
from sklearn.metrics import classification_report
from sklearn.preprocessing import robust_scale
from sklearn import preprocessing

project_dir = '/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/'
input_file = '/Users/ishwarya/Documents/ITIS_Sem_II/DataMiningI/Classification_project/data.csv'

# comma delimited is the default
#df = pd.read_csv(input_file, header=0)
#df_target = pd.read_csv(input_file, header=0, usecols=[10])

dataset=pd.read_csv(input_file, sep=',',names = ["fLength", "fWidth", "fSize", "fConc","fConc1","fAsym","fM3Long","fM3Trans","fAlpha","fDist","class"])

# create design matrix X and target vector y
X = np.array(dataset.ix[:, 0:10])   # end index is exclusive
y = np.array(dataset['class'])  # another way of indexing a pandas df

scaled_x=robust_scale(X, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
#print("Scaled data",scaled_x)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Splitting scaled dataset
scaled_x_train, scaled_x_test, scaled_y_train, scaled_y_test = train_test_split(scaled_x, y, test_size=0.33, random_state=42)

pred_values=[]
acc_values=[]

scaled_pred_values=[]
scaled_acc_values=[]
#kDict ={}

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

def knn_classifier(kValue):
    # Applying Classifier
    clf = None
    if (kValue == 1):
        print ("For K=1 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 2):
        print ("For K=2 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 3):
        print ("For K=3 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 4):
        print ("For K=4 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 5):
        print ("For K=5 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 6):
        print ("For K=6 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 7):
        print ("For K=7 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 8):
        print ("For K=8 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 9):
        print ("For K=9 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 10):
        print ("For K=10 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 11):
        print ("For K=11 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    acc_values.append(accuracy)
    pred_values.append(y_pred)
    print ('KNN classifier prediction for K=',kValue,':', y_pred)
    print('Accuracy Score for K=',kValue,':', accuracy,'\n')

def knn_classifier_scaled(kValue):
    # Applying Classifier
    clf = None
    if (kValue == 1):
        print ("For K=1 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 2):
        print ("For K=2 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 3):
        print ("For K=3 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 4):
        print ("For K=4 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 5):
        print ("For K=5 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 6):
        print ("For K=6 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 7):
        print ("For K=7 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 8):
        print ("For K=8 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 9):
        print ("For K=9 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 10):
        print ("For K=10 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    if (kValue == 11):
        print ("For K=11 Neighbor:")
        clf = KNeighborsClassifier(n_neighbors=kValue,metric='manhattan',weights='distance')
    clf = clf.fit(scaled_x_train, scaled_y_train)
    scaled_y_pred = clf.predict(scaled_x_test)
    scaled_accuracy = accuracy_score(scaled_y_test, scaled_y_pred)
    scaled_acc_values.append(scaled_accuracy)
    scaled_pred_values.append(scaled_y_pred)
    print ('KNN classifier prediction for K=',kValue,':', scaled_y_pred)
    print('Accuracy Score for K=',kValue,':', scaled_accuracy,'\n')
    return [scaled_accuracy,scaled_y_pred,clf]

#Running Cases
#Build KNN classifier for K value 1 to 11
print("===============================================================================")
print("Classifier prediction without data preprocessing:\n")
print("===============================================================================")
for k in range(1, 12):
	knn_classifier(k)
print("===============================================================================")
print("Classifier prediction after data preprocessing - After applying robust scaling:\n")
print("===============================================================================")
for k in range(1, 12):
	result_knn=knn_classifier_scaled(k)

def evaluationMeasures(classifier_type):
	print("===============================================================================")
	print("\nEvaluation Measures for",classifier_type,":")
	print("===============================================================================")
	print("Classification Report:\n\n",classification_report(y_test, y_pred_final))

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

#ROC and AUC
def roc_and_auc(kvalue):
	clf = KNeighborsClassifier(n_neighbors=kvalue,metric='manhattan',weights='distance')
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
	plt.title('ROC curve for the KNN classifier')
	plt.grid(True)
	plt.savefig(project_dir+'roc_curve_knn.png')
	print("===============================================================================")
	#AUC score for SVM
	print ('This is the AUC for the KNN classifier')
	print (metrics.roc_auc_score(y_test_bin,prob2))

# Comparing accuracy
print("===============================================================================")
print ('Comparing Accuracy for different values of K:')

temp_acc = acc_values[0]
total = len(acc_values)
for i,item in enumerate(acc_values):
    if (i < total-2):
        if acc_values[i+1] > temp_acc:
            temp_acc = acc_values[i+1]
            kVal=i+2
    if (i == total-1):
        if acc_values[i] > temp_acc:
            temp_acc = acc_values[i]
            kVal=i+1

y_pred_final =pred_values[kVal-1]
print("Highest accuracy score and its K value (without data preprocessing):", temp_acc,",",kVal)

scaled_temp_acc = scaled_acc_values[0]
scaled_total = len(scaled_acc_values)

for i2,item2 in enumerate(scaled_acc_values):
    if (i2 < scaled_total-2):
        if scaled_acc_values[i2+1] > scaled_temp_acc:
            scaled_temp_acc = scaled_acc_values[i2+1]
            scaled_kVal=i2+2
    if (i2 == scaled_total-1):
        if scaled_acc_values[i2] > scaled_temp_acc:
            scaled_temp_acc = scaled_acc_values[i2]
            scaled_kVal=i2+1

scaled_y_pred_final =scaled_pred_values[scaled_kVal-1]
print("Highest accuracy score and its K value (with preprocessed robust-scaled data):", scaled_temp_acc,",",scaled_kVal)

#Calculate Confusion Matrix
if scaled_temp_acc > temp_acc:
    confusionMatrix =calcuate_confusionMatrix('KNN', scaled_y_pred_final)
    evaluationMeasures("KNN")
    roc_and_auc(scaled_kVal)
else:
    confusionMatrix =calcuate_confusionMatrix('KNN',y_pred_final)
    evaluationMeasures("KNN")
    roc_and_auc(kVal)












