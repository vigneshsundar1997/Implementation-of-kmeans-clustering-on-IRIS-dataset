from os import PRIO_PGRP
from numpy import pi, random
from numpy.core.fromnumeric import mean
from numpy.core.numeric import count_nonzero
from numpy.lib.function_base import diff
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
import timeit
import math
from sys import argv

np.random.seed(0)

#kmeans function computing the cluster values, WC-SSD, SC and NMI
def kmeans(digits_embedded_array,k):
    #randomnly chose the initial k centroids
    centroid_indexes = np.random.choice(digits_embedded_array.shape[0], k, replace = False)

    #get the x and y column
    digits_embedded_array_xy = digits_embedded_array[:,2:4]

    #get the centroid values from the chosen indexes
    centroid_array = digits_embedded_array_xy[centroid_indexes,:]

    #train the model
    for iter in range(50):
        distances = cdist(digits_embedded_array_xy, centroid_array ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
        if iter<49:
            temp_array = np.zeros(shape=(k,2))
            for i in range(k):
                temp_array[i] = np.mean(digits_embedded_array_xy[np.where(points == i)],axis=0)
            if np.array_equal(temp_array, centroid_array):
                break
            else:
                centroid_array = temp_array
                    
    #calculate pair wise distances of all the points to avoid recomputing everytime
    pair_wise_distances = cdist(digits_embedded_array_xy, digits_embedded_array_xy ,'euclidean')
    

    clusterIndexes = {}
    clusterEntropy = 0
    labelEntropy = 0

    classEntropy = 0
    #calculate class label entropy
    for digit in np.unique(digits_embedded_array[:,1]):
        digit_class = digits_embedded_array[np.where(digits_embedded_array[:,1]==digit)]
        ent = digit_class.shape[0] / digits_embedded_array.shape[0]
        classEntropy += (-ent * math.log(ent))

    #calculate cluster entropy and label cluster conditional entropy and also store the index list of each clusters
    for cluster in range(k):
        #store the indexlist of each cluster
        indexList = np.array(np.where(points==cluster)).flatten().tolist()
        #dictionary to store the indexes
        clusterIndexes[cluster] = indexList
        ent = len(indexList)/points.shape[0]
        clusterEntropy += (-ent * math.log(ent))

        digits_array = digits_embedded_array[indexList]
        #conditional label cluster entropy
        labelEntropyCluster = 0
        for label in np.unique(digits_embedded_array[:,1]):
            labelCount = digits_array[np.where(digits_array[:,1]==label)].shape[0]
            if labelCount!=0:
                labelEnt = labelCount/digits_array.shape[0]
                labelEntropyCluster += (-labelEnt * math.log(labelEnt))
        labelEntropy += ent * labelEntropyCluster

    mutual_information = classEntropy - labelEntropy

    nmi = mutual_information / (classEntropy + clusterEntropy)

    #calculate within cluster distance
    wc_ssd = 0

    for digit in range(k):
        digits_array = digits_embedded_array_xy[clusterIndexes[digit]]
        intra_distance = cdist(digits_array, centroid_array[digit,:].reshape(1,2) ,'euclidean')
        wc_ssd += np.sum(np.square(intra_distance))

    #calculate silhoutee coefficient
    silhoutte = 0

    for cluster in range(k):
        sameCluster = clusterIndexes[cluster]
        A_index = pair_wise_distances[sameCluster,:][:,sameCluster]
        #calculate the A value for all the points in a cluster
        if A_index.shape[0]==1:
            A_mean = 0
        else:
            A_mean = np.sum(A_index,axis=1)/(A_index.shape[0]-1)
        iter = 0
        #calculate the B value for all the points in a cluster
        for diffCluster in range(k):
            if diffCluster!=cluster:
                indexes = clusterIndexes[diffCluster]
                B_index = pair_wise_distances[sameCluster,:][:,indexes]
                tempB_mean = np.mean(B_index,axis=1)
                if iter == 0:
                    B_mean = tempB_mean
                else:
                    B_mean = np.minimum(B_mean,tempB_mean)
                iter = iter+1
        maxAB = np.maximum(A_mean,B_mean)
        silhoutte += np.sum(np.divide(np.subtract(B_mean,A_mean),maxAB))

    silhoutte = silhoutte/digits_embedded_array.shape[0]
    return wc_ssd,silhoutte,nmi

if __name__=="__main__":
    dataFileName = argv[1]
    k = int(argv[2])

    data_embedded = pd.read_csv(dataFileName,header=None)
    digits_embedded_array = data_embedded.to_numpy()
    wc_ssd,silhoutte,nmi = kmeans(digits_embedded_array,k)

    print('WC-SSD:',wc_ssd)
    print('SC:', silhoutte)
    print('NMI:', nmi)