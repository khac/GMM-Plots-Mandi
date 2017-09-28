'''
Created By: Adit Chawdhary 11 June,2017
This file will take the test and train data and the mixture models and will plot the contour plots as well as the decision boundary
'''

import sys
import _pickle as cPickle
import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture
import os.path
import matplotlib.pyplot as plt
import matplotlib as mpl

n_components = int(sys.argv[1])

colors_2 = ['orange', 'green']
colors_4 = ['orange', 'green', 'blue','orange']
colors_8 = ['orange', 'green', 'blue','orange','gold','silver','pink','yellow']
colors_16 = ['orange', 'green', 'blue','orange','gold','silver','pink','yellow','black','brown','violet','grey','lightgreen','olive','indigo','fuchsia']
colors_32 = ['orange', 'green', 'blue','orange','gold','silver','pink','yellow','black','brown','violet','grey','lightgreen','olive','indigo','fuchsia','red', 'blue', 'green','orange','gold','silver','pink','yellow','black','brown','violet','grey','lightgreen','olive','indigo','fuchsia']
colors_64 = ['orange', 'green', 'blue','orange','gold','silver','pink','yellow','black','brown','violet','grey','lightgreen','olive','indigo','fuchsia','red', 'blue', 'green','orange','gold','silver','pink','yellow','black','brown','violet','grey','lightgreen','olive','indigo','fuchsia','red', 'blue', 'green','orange','gold','silver','pink','yellow','black','brown','violet','grey','lightgreen','olive','indigo','fuchsia','red', 'blue', 'green','orange','gold','silver','pink','yellow','black','brown','violet','grey','lightgreen','olive','indigo','fuchsia']
clr = [['fuchsia'],['fuchsia','fuchsia'],['fuchsia','fuchsia','fuchsia','fuchsia'],['fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia'],['fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia'],['fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia'],['fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia'],['fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia','fuchsia']]
color_train= ['black','0.1','brown']

if int(n_components) == 2:
	colors = colors_2
elif int(n_components) == 4:
	colors = colors_4
elif int(n_components) == 8:
	colors = colors_8
elif int(n_components) == 16:
	colors = colors_16
elif int(n_components) == 32:
	colors = colors_32
elif int(n_components) == 64:
	colors = colors_64


def make_ellipses(gmm, ax):
    for n, color in enumerate(clr[n_components-1]):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

path = '/home/adit/Desktop/Predicted_Data'
num_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
print("Predicted result are being retrieved")

if num_files==0:
	print("No Predicted classes found, test the models using Test_file.py")
	exit()

training_data,i = [],1
path = '/home/adit/Desktop/Training_Data/Train'
num_training_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
while i <= num_training_files:
	training_data.append(cPickle.load(open("/home/adit/Desktop/Training_Data/Train/Training_data_"+str(i)+".p", 'rb')))
	i += 1

models,i = [],1
path = '/home/adit/Desktop/Training_Data/Models'
num_training_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
while i <= num_training_files/2:
	for j in ['diag','full']:
		models.append(cPickle.load(open("/home/adit/Desktop/Training_Data/Models/save_class_"+str(i)+"_cov_"+j+".p", 'rb')))
	i += 1

test_data,i = [],1
path = '/home/adit/Desktop/Training_Data/Test'
num_test_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
while i <= num_test_files:
	test_data.append(cPickle.load(open("/home/adit/Desktop/Training_Data/Test/Test_data_"+str(i)+".p", 'rb')))
	i += 1



#print(models[0],'\n\n',models[1])
f,i,j=0,0,0
while f < len(models):
	if f%2 == 1:
		h = plt.subplot(2,1,1)
		make_ellipses(models[f], h)
		#plt.title("Decision Boundary, Covariance Type Diagonal")
		plt.xticks()
		plt.yticks()


	elif f%2 == 0:
		h = plt.subplot(2,1,2)
		make_ellipses(models[f], h)	
		#plt.title("Decision Boundary, Covariance Type Full")
		plt.xticks()
		plt.yticks()

	f += 1		


total_data = training_data+test_data

for i in total_data:
	i = np.array(i)
	x_min,x_max,y_max,y_min = 0.0,0.0,0.0,0.0
	if x_max < max(i[:,0]):
		x_max = max(i[:,0])
	if x_min > min(i[:,0]):
		x_min = min(i[:,0])
	if y_max < max(i[:,1]):
		y_max = max(i[:,1])
	if y_min > min(i[:,1]):
		y_min = min(i[:,1])

x_range, y_range = 200,200
gx=np.linspace(-3,3,x_range)
gy=np.linspace(-3,3,y_range)
global1 = []

for i in range(x_range):
    for j in range(y_range):
        global1.append([gx[i],gy[j]])
global1 = np.array(global1)

models = models[::2]

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

predictions_for_decision_boundary,j = [],0
for i in global1:
	likelyhood_for_all_classes = []
	for gmm in models:
		likelyhood_for_all_classes.append(class_likelyhood(gmm,[i],'diag',n_components))
	predictions_for_decision_boundary.append(likelyhood_for_all_classes.index(max(likelyhood_for_all_classes)))
	plt.figure(1)
	plt.subplot(2,1,1)
	#plt.title("Desision Boundary")
	plt.scatter(i[ 0], i[1], marker='o', alpha = 0.5 ,color=colors_64[predictions_for_decision_boundary[j]])
	plt.subplot(2,1,2)
	plt.scatter(i[0],i[1],marker= "o",alpha =0.5, color= colors_64[predictions_for_decision_boundary[j]] )
	j += 1


num_training_files = sum(os.path.isfile(os.path.join(path, f)) for f in os.listdir(path))
i,j = 0,1
while j <= 2:
	k = 0 
	for i in range(len(training_data)):
		plt.subplot(2,1,j)
		plot_training_data_per_class = np.array(training_data[i])
		plt.scatter(plot_training_data_per_class[:, 0], plot_training_data_per_class[:, 1], marker='.' ,color=color_train[k])
		plt.xticks()
		plt.yticks()
		k += 1
	j += 1
	

plt.show()

