'''
Created By: Adit Chawdhary 10 June, 2017
This file will take the previously trained models and then print confusion matrix and the Accuracy for the given test data.
'''
import sys
import _pickle as cPickle
import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture
import os.path

n_components = int(sys.argv[1])
path = '/home/adit/Desktop/Training_Data/Models'
num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
print("Total Gaussian Mixture Models found are for ",num_files/2," classes")

if num_files==0:
	print("No Trained Models found, train the models using Train_file.py")
	exit()

cov_type = ['diag','full']
models = []
test_data = []
labels = []
predicted_class = []
i = 1
while i <= num_files/2:
	all_models_in_a_class = []
	for j in cov_type:
		all_models_in_a_class.append(cPickle.load(open("/home/adit/Desktop/Training_Data/Models/save_class_"+str(i)+"_cov_"+j+".p", 'rb')))
	models.append(all_models_in_a_class)
	test_data.append(cPickle.load(open("/home/adit/Desktop/Training_Data/Test/Test_data_"+str(i)+".p", 'rb')))
	labels.append(cPickle.load(open( "/home/adit/Desktop/Training_Data/Labels/Test_label"+str(i)+".p", "rb" )))
	i +=1
'''
The step above will retrieve the stored Gaussian Mixture Models, the Test data and the Data Labels into the lists named models, 
test_data and labels.
'''
def class_likelyhood(gmm,test_data,covariance_type,n_components):
    llk = []
    if covariance_type == 'full':
        for i in test_data:
            prob = 0.0
            for j in range(n_components):
                covariance = gmm[1].covariances_[j]
                #print(np.shape(covariance))
                covariance = np.matrix(covariance[:2])
                #print(np.shape(covariance))
                #covariance = covariance.reshape((2,2))
                determinant =  np.linalg.det(covariance)
                mean = gmm[1].means_[j]
                weight = gmm[1].weights_[j]
                denominator = 2*np.pi*np.sqrt(determinant)
                numerator =np.exp(-0.5*np.dot(np.dot((i-mean).T,np.linalg.inv(covariance)),(i-mean)))*weight
        
                likelhood = numerator/denominator
                #print(i,"\n",covariance,"\n",determinant,"\n",mean,"\n",weight,"\n",denominator,"\n",numerator,"\n",likelhood,"\n\n")

                prob += likelhood
                #print(np.asscalar(np.log(prob)))
            llk.append(np.asscalar(np.log(prob)))   
    
    elif covariance_type == 'diag':
        for i in test_data:
            prob = 0.0
            for j in range(n_components):
                covariance = gmm.covariances_[j]
                #print(np.shape(covariance))
                covariance = np.multiply(np.eye(2),gmm.covariances_[j][:2])
                #print(np.shape(covariance))
                #covariance = covariance.reshape((2,2))
                determinant =  np.linalg.det(covariance)
                mean = gmm.means_[j]
                weight = gmm.weights_[j]
                denominator = 2*np.pi*np.sqrt(determinant)
                numerator =np.exp(-0.5*np.dot(np.dot((i-mean).T,np.linalg.inv(covariance)),(i-mean)))*weight
                likelhood = numerator/denominator
                #print(i,"\n",covariance,"\n",determinant,"\n",mean,"\n",weight,"\n",denominator,"\n",numerator,"\n",likelhood,"\n\n")
                prob += likelhood
            llk.append(np.asscalar(np.log(prob)))
    return llk

'''

Now, for the number of test tuples in each class a likelyhood for each class will be calculated and whichever gives highest likelyhood 
that tuple will be allotted to that test tuple will be stored as the predicted class.


print(len(test_data[0]),len(test_data[1]),len(test_data[2]))
In the data sets the number of data tuples may not exactly equal, the slight variation can be seen in the confusion matrix.
'''
for i in range(len(models)):
	models[i] = models[i][0]


i,j = 1,0
while i<=len(test_data):
	likelyhoods_for_every_class = []
	predictions_for_one_class = []
	for test_data_tuple in test_data[i-1]:
		likelyhood_for_each_tuple = []
		for gmm in models:
			likelyhood_for_each_tuple.append(class_likelyhood(gmm,[test_data_tuple],'diag',n_components))
		predictions_for_one_class.append(likelyhood_for_each_tuple.index(max(likelyhood_for_each_tuple))+1)
	predicted_class.append(predictions_for_one_class)
	i += 1

#print(labels,predicted_class)
j,i,confusion_matrix = 0,0,np.zeros((len(test_data),len(test_data)))
for j in range(len(labels)):
	for i in range(len(predicted_class)):
		confusion_matrix[i][j] = np.count_nonzero(np.array(predicted_class[i]) == labels[j][0])

#print((np.array(predicted_class[0])==1).sum(),(np.array(predicted_class[0])==2).sum(),(np.array(predicted_class[0])==3).sum())

if len(labels)==2:
	print("__________Confusion Matrix__________\n\t     Predicted Class\nActual Class",confusion_matrix[0],"\n\t    ",confusion_matrix[1])
elif len(labels)==3:
	print("__________Confusion Matrix__________\n\t       Predicted Class\nActual Class",confusion_matrix[0],"\n\t    ",confusion_matrix[1],"\n\t    ",confusion_matrix[2])
else:
	print(confusion_matrix)
correct,total,i = 0.0,0.0,0
for i in range(len(labels)):
	correct += confusion_matrix[i][i]
	total += confusion_matrix[i].sum()
print("Accuracy: ",correct/total )
cPickle.dump( predicted_class, open( "/home/adit/Desktop/Predicted_Data/Predicted_class.p", "wb" ) )
	