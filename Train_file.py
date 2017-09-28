'''
Created By: Adit Chawdhary 9 June, 2017
Here Gaussian Mixture Models will be created for each class and this type of distribution is called multi-modal distribution.
The train data will be stored in the path variable below, a folder named TrainingFiles on the Desktop.
And the number of mixtures per model will be decided by the user.

Also each class has to be named Class1, Class2, Class3 and so on. Note the capital C in the beginning

When running the script in the terminal pass the argument for the number of mixtures per model. For example, python Train_file.py 4  
'''
import sys
import _pickle as cPickle
import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture
import os.path
path = '/home/adit/Desktop/TrainingFiles'
num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
print("Total number of classes are: ",num_files)

if num_files==0:
	print("No Files in the defined directory")
	exit()

#a = [[[1,2],[2,3],[4,5]],[[6,7],[8,9],[10,11]],[[12,13],[14,15],[16,17]],[[18,19],[20,21],[22,23]]]
#print(a[1])

training_data = []
labels = []
i=1
n_components = int(sys.argv[1])
while i <= num_files:
	all_models_in_a_class=[]
	all_data_tuple_in_a_class = []
	all_data_labels_in_a_class = []
	a = open('/home/adit/Desktop/TrainingFiles/Class'+str(i)+'.txt')
	for j in a:
		each_data_tuple_in_a_class = []
		j = j.split(" ")
		each_data_tuple_in_a_class.append(float(j[0]))
		each_data_tuple_in_a_class.append(float(j[1]))
		all_data_labels_in_a_class.append(i)
		all_data_tuple_in_a_class.append(each_data_tuple_in_a_class)
	training_data.append(all_data_tuple_in_a_class)
	labels.append(all_data_labels_in_a_class)
	#print(training_data,"\n\n",labels)
	X_train = training_data[i-1][:int(0.75*len(training_data[i-1]))]
	X_test = training_data[i-1][int(0.75*len(training_data[i-1])):]
	y_train = labels[i-1][:int(0.75*len(labels[i-1]))]
	y_test = labels[i-1][int(0.75*len(labels[i-1])):]
	cPickle.dump( X_test, open( "/home/adit/Desktop/Training_Data/Test/Test_data_"+str(i)+".p", "wb" ) )
	cPickle.dump( y_test, open( "/home/adit/Desktop/Training_Data/Labels/Test_label"+str(i)+".p", "wb" ) )
	cPickle.dump( X_train, open( "/home/adit/Desktop/Training_Data/Train/Training_data_"+str(i)+".p", "wb" ) )

	estimators = dict((cov_type, GaussianMixture(n_components=n_components,covariance_type=cov_type, max_iter=10000, random_state=0,tol=1e-3)) for cov_type in [ 'diag', 'full'])
	n_estimators = 2 # one for diagonal covariance and the other for full covariance
	for index, (name, estimator) in enumerate(estimators.items()):
		all_models_in_a_class.append(estimator.fit(X_train))
		cPickle.dump( all_models_in_a_class[index], open( "/home/adit/Desktop/Training_Data/Models/save_class_"+str(i)+"_cov_"+name+".p", "wb" ) )
	#print(all_models_in_a_class,"\n")
	i +=1	

print("Training the GMM for the ",num_files," classes is complete.")

