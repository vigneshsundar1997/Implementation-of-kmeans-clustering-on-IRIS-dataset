from datetime import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
import math
from sys import argv

np.random.seed(0)

def kmeans(digits_embedded_array,k,isWCCNeeded,isSCNeeded,isPointsNeeded):
    centroid_indexes = np.random.choice(digits_embedded_array.shape[0], k, replace = False)

    digits_embedded_array_xy = digits_embedded_array[:,2:4]

    centroid_array = digits_embedded_array_xy[centroid_indexes,:]

    classEntropy = 0

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

    #calculate pair wise distances of all the points
    pair_wise_distances = cdist(digits_embedded_array_xy, digits_embedded_array_xy ,'euclidean')
    

    clusterIndexes = {}
    clusterEntropy = 0
    labelEntropy = 0

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
    if isWCCNeeded:
        for digit in range(k):
            digits_array = digits_embedded_array_xy[clusterIndexes[digit]]
            intra_distance = cdist(digits_array, centroid_array[digit,:].reshape(1,2) ,'euclidean')
            wc_ssd += np.sum(np.square(intra_distance))

    #calculate silhoutee coefficient
    silhoutte = 0
    if isSCNeeded:
        for cluster in range(k):
            sameCluster = clusterIndexes[cluster]
            A_index = pair_wise_distances[sameCluster,:][:,sameCluster]
            #calculate the A value for all the points in a cluster
            if A_index.shape[0]==1:
                A_mean = 0
            else:
                A_mean = np.sum(A_index,axis=1)/(A_index.shape[0]-1)
            iter = 0
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
    
    if isPointsNeeded:
        return wc_ssd,silhoutte,nmi,points
    return wc_ssd,silhoutte,nmi

def kanalysis(digits_embedded_array):
    dataset1 = digits_embedded_array
    dataset2 = digits_embedded_array[np.where((digits_embedded_array[:,1]==2) | (digits_embedded_array[:,1]==4) | (digits_embedded_array[:,1]==6) | (digits_embedded_array[:,1]==7))]
    dataset3 = digits_embedded_array[np.where((digits_embedded_array[:,1]==6) | (digits_embedded_array[:,1]==7))]
    datasetLists = []

    datasetLists.append(dataset1)
    datasetLists.append(dataset2)
    datasetLists.append(dataset3)

    k = [2,4,8,16,32]

    datasetWCC = []
    
    datasetSC = []

    wccList = []
    scList = []
    i=1
    for dataset in datasetLists:
        for cluster in k:
            wcc,sc,nmi = kmeans(dataset,cluster,True,True,False)

            datasetWCC.append(wcc)
            datasetSC.append(sc)

        wccList.append(datasetWCC)
        scList.append(datasetSC)

        print('WC-SSD for Dataset' , i , ':' , datasetWCC)
        print('SC for Dataset' , i , ':' , datasetSC)
        datasetWCC = []
        datasetSC = []
        i=i+1

    plt.plot(k,wccList[0])
    plt.xlabel("K values")
    plt.ylabel("WC-SSD")
    plt.title("WC-SSD as a function of K for dataset 1")
    plt.show()

    plt.plot(k,scList[0])
    plt.xlabel("K values")
    plt.ylabel("Silhoutte Coefficient(SC)")
    plt.title("SC as a function of K for dataset 1")
    plt.show()

    plt.plot(k,wccList[1])
    plt.xlabel("K values")
    plt.ylabel("WC-SSD")
    plt.title("WC-SSD as a function of K for dataset 2")
    plt.show()

    plt.plot(k,scList[1])
    plt.xlabel("K values")
    plt.ylabel("Silhoutte Coefficient(SC)")
    plt.title("SC as a function of K for dataset 2")
    plt.show()

    plt.plot(k,wccList[2])
    plt.xlabel("K values")
    plt.ylabel("WC-SSD")
    plt.title("WC-SSD as a function of K for dataset 3")
    plt.show()

    plt.plot(k,scList[2])
    plt.xlabel("K values")
    plt.ylabel("Silhoutte Coefficient(SC)")
    plt.title("SC as a function of K for dataset 3")
    plt.show()

#perform analysis on the seed value's impact
def seedanalysis(digits_embedded_array):
    dataset1 = digits_embedded_array
    dataset2 = digits_embedded_array[np.where((digits_embedded_array[:,1]==2) | (digits_embedded_array[:,1]==4) | (digits_embedded_array[:,1]==6) | (digits_embedded_array[:,1]==7))]
    dataset3 = digits_embedded_array[np.where((digits_embedded_array[:,1]==6) | (digits_embedded_array[:,1]==7))]
    
    datasetLists = []

    datasetLists.append(dataset1)
    datasetLists.append(dataset2)
    datasetLists.append(dataset3)

    datasetWCCAverage = []
    datasetSCAverage = []

    datasetWCCSD = []
    datasetSCSD = []

    wccListAverage = []
    scListAverage = []

    wccListSD = []
    scListSD = []

    seeds = np.random.choice(30,size=10,replace=False)

    k = [2,4,8,16,32]

    datasetWCCSeed = []
    datasetSCSeed = []

    for i in range(len(datasetLists)):
        for cluster in k:
            print("Calculating WC-SSD and SC for k", cluster , 'for dataset' , (i+1))
            for x in seeds:
                np.random.seed(x)
                wcc,sc,nmi = kmeans(datasetLists[i],cluster,True,True,False)
                datasetWCCSeed.append(wcc)
                datasetSCSeed.append(sc)
            
            print('Average WC-SSD value for dataset ' + str(i+1) + ' with k ' + str(cluster) + ': ' + str(np.mean(datasetWCCSeed)))
            print('Average SC value for dataset ' + str(i+1) + ' with k ' + str(cluster) + ': ' + str(np.mean(datasetSCSeed)))

            datasetWCCAverage.append(np.mean(datasetWCCSeed))
            datasetSCAverage.append(np.mean(datasetSCSeed))

            print('SD of WC-SSD value for dataset ' + str(i+1) + ' with k ' + str(cluster) + ': ' + str(np.std(datasetWCCSeed)))
            print('SD of SC value for dataset ' + str(i+1) + ' with k ' + str(cluster) + ': ' + str(np.std(datasetSCSeed)))

            print('--------------------------------')
            
            datasetWCCSD.append(np.std(datasetWCCSeed))
            datasetSCSD.append(np.std(datasetSCSeed))

            datasetWCCSeed=[]
            datasetSCSeed=[]

        wccListAverage.append(datasetWCCAverage)
        scListAverage.append(datasetSCAverage)

        wccListSD.append(datasetWCCSD)
        scListSD.append(datasetSCSD)

        datasetWCCAverage=[]
        datasetSCAverage=[]
        datasetWCCSD=[]
        datasetSCSD=[]

    plt.errorbar( k, wccListAverage[0], yerr=wccListSD[0] ,label='Dataset 1 WC-SSD')
    plt.xlabel("K values")
    plt.ylabel("WC-SSD")
    plt.title("Mean of WC-SSD for ten seeds and SD as a function of K for dataset 1")
    plt.show()

    plt.errorbar( k, scListAverage[0], yerr= scListSD[0] ,label='Dataset 1 SC')
    plt.xlabel("K values")
    plt.ylabel("Silhoutte Coefficient(SC)")
    plt.title("Mean of SC for ten seeds and SD as a function of K for dataset 1")
    plt.show()

    plt.errorbar( k, wccListAverage[1], yerr= wccListSD[1] ,label='Dataset 2 WC-SSD')
    plt.xlabel("K values")
    plt.ylabel("WC-SSD")
    plt.title("Mean of WC-SSD for ten seeds and SD as a function of K for dataset 2")
    plt.show()

    plt.errorbar( k, scListAverage[1], yerr= scListSD[1] ,label='Dataset 2 SC')
    plt.xlabel("K values")
    plt.ylabel("Silhoutte Coefficient(SC)")
    plt.title("Mean of SC for ten seeds and SD as a function of K for dataset 2")
    plt.show()

    plt.errorbar( k, wccListAverage[2], yerr= wccListSD[2] ,label='Dataset 3 WC-SSD')
    plt.xlabel("K values")
    plt.ylabel("WC-SSD")
    plt.title("Mean of WC-SSD for ten seeds and SD as a function of K for dataset 3")
    plt.show()

    plt.errorbar( k, scListAverage[2], yerr= scListSD[2] ,label='Dataset 3 SC')
    plt.xlabel("K values")
    plt.ylabel("Silhoutte Coefficient(SC)")
    plt.title("Mean of SC for ten seeds and SD as a function of K for dataset 3")
    plt.show()

#visualize the points
def visualize(dataset,points,number):
    indices = np.random.randint(0,dataset.shape[0],size=1000)
    dataset_xy = dataset[:,2:]   
    clusters = points[indices]
    x = dataset_xy[indices,:][:,0]
    y = dataset_xy[indices,:][:,1]
    plt.figure(figsize=(10, 8))
    for cluster in np.unique(clusters):
        labelIndex = np.where(clusters==cluster)[0]
        plt.scatter(x[labelIndex],y[labelIndex],label=cluster)
    plt.legend()
    title = 'Visualization of 1000 data points for Dataset ' + str(number)
    plt.title(title)
    plt.show()

#calculate nmi and do visualization
def nmiAndVisualization(digits_embedded_array):
    dataset1 = digits_embedded_array
    dataset2 = digits_embedded_array[np.where((digits_embedded_array[:,1]==2) | (digits_embedded_array[:,1]==4) | (digits_embedded_array[:,1]==6) | (digits_embedded_array[:,1]==7))]
    dataset3 = digits_embedded_array[np.where((digits_embedded_array[:,1]==6) | (digits_embedded_array[:,1]==7))]

    wcc,sc,nmi,points = kmeans(dataset1,8,False,False,True)
    print('NMI for dataset 1 with k=8:' , nmi)
    visualize(dataset1,points,1)

    wcc,sc,nmi,points = kmeans(dataset2,4,False,False,True)
    print('NMI for dataset 2 with k=4:', nmi)
    visualize(dataset2,points,2)

    wcc,sc,nmi,points = kmeans(dataset3,2,False,False,True)
    print('NMI for dataset 2 with k=2:',nmi)
    visualize(dataset3,points,3)

if __name__=="__main__":
    data_embedded = pd.read_csv("digits-embedding.csv",header=None)
    digits_embedded_array = data_embedded.to_numpy()
    option = argv[1]
    if option == '1':
        kanalysis(digits_embedded_array)
    if option == '2':
        seedanalysis(digits_embedded_array)
    if option == '3':
        nmiAndVisualization(digits_embedded_array)