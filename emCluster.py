#Jack Ziegler
#EM cluster breast cancer

import numpy as np
import sys
import csv


k = 2
data = np.loadtxt('breastCancerEM.csv', delimiter=",", usecols=(0,1,2,3,4,5,6,7,8))
n, d = data.shape
classes = np.loadtxt('breastCancerEM.csv', delimiter=",", dtype=np.str, usecols=(9))
for i in range (len(classes)):
    if (classes[i] == 2):
        classes[i] = 0
    else:
        classes[i] = 1        
classes = classes.astype(np.float)

def EM(data, k, means, covs, priors, offset=0.00001):
    n, d = data.shape    
    oldMeans = np.zeros((k,d))    
    iterNum = 1

    while True:
        wij = np.zeros((k,n))
        for i in range(k):
            for j in range(n):
                temp = 0
                for m in range(k):
                    temp = temp + probPart(data[j,:],means[m,:],covs[m][0]) * priors[m]
                wij[i,j] = probPart(data[j,:],means[i,:],covs[i][0]) * priors[i] / temp
        for i in range(k):
            means[i,:] = 0
            for j in range(n):
                means[i,:] = means[i,:] + wij[i,j] * data[j,:] 
            means[i,:] = means[i,:] / np.sum(wij[i,:],axis=0)
            covs[i][0] = np.zeros((d,d))
            for m in range(d):
                for h in range(d):
                    for j in range(n):
                        covs[i][0][m][h] = covs[i][0][m][h] + wij[i,j] * (data[j,m] - means[i,m]) * (data[j,h] - means[i,h])
            covs[i][0] = covs[i][0] / np.sum(wij[i,:],axis=0)
            priors[i] = np.sum(wij[i,:],axis=0) / n
        if (np.linalg.pinv(means - oldMeans) ** 2 <= offset).any():
            clusters = [[] for i in range(k)]
            labels = [[] for i in range(k)]
            clusterLabel = np.zeros((n))
            for i in range(n):
                maxProb = sys.float_info.min
                maxIdx = -1
                for j in range(k):
                    prob = probPart(data[i,:],means[j,:],covs[j][0])
                    if prob > maxProb:
                        maxProb = prob
                        maxIdx = j
                        clusterLabel[i] = maxIdx
                clusters[maxIdx].append(data[i,:])
                labels[maxIdx].append(i+1)
                
            return clusters, labels, clusterLabel, means, covs, priors, iterNum
        iterNum = iterNum + 1
        oldMeans = np.copy(means)

def probPart(x, means, covs):
    return 1. / (((2*np.pi) ** (float(covs.shape[0])/2)) * (np.linalg.det(covs) ** (1./2))) * np.exp(-(1./2) * (x-means).T @ np.linalg.inv(covs) @ (x-means))


#Main
cluster1 = data[0:50]
muC1 = cluster1.mean(axis=0)
cluster2 = data[50:100]
muC2 = cluster2.mean(axis=0)
means = np.vstack((muC1,muC2))
covs = [[] for i in range(k)]
for i in range(k):
    covs[i].append(np.identity(d))     
priors = np.ones((k,1)) * (1./k)

clusters, labels, clusterLabel, final_means, final_covs, final_priors, iterNum = EM(data, k, means, covs, priors)
clusterLabel = clusterLabel.astype(np.float)

print(final_means)


#Main
tp = 0
tn = 0
fp = 0
fn = 0
cluster1 = 0
cluster2 = 0
count = 0
c1 = labels[0]
c2 = labels[1]
classes = []
with open('breastCancerEM.csv') as csvfile:
    csvReader = csv.reader(csvfile)
    for row in csvReader:
        classes.append(row[9])
for i in classes:
    count += 1
    
    if i == '2':
        exists = count in c1
        if exists == True:
            tn += 1
        else:
            fn += 1
        
    else:
        exists = count in c2
        if exists == True:
            tp += 1
        else:
            tn += 1
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+fn+tn+fp)
balancedAccuracy = (specificity+sensitivity)/2
total = 699
print("")
print("Number of data points in Cluster 1: ", len(labels[0]))
print("Number of data points in Cluster 2: ", len(labels[1]))
print("")
print("Specificity is: ", round(float(specificity),2))
print("Sensitivity is: ", round(float(sensitivity),2))
print("Balanced Accuracy is: ", round(float(balancedAccuracy),2))
print("Accuracy is: ", round(float(accuracy),2))
clusterss=[]
for i in range (len(clusterLabel)):
    if clusterLabel[i] ==0:
        clusterss.append('2')
    else:
        clusterss.append('4')
np.savetxt("Em_clusters.txt",np.array(clusterss), fmt = "%s")

print(means)