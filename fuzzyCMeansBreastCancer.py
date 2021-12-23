#Jack Ziegler Fuzzy-C
import pandas as pd
import numpy as np
import random
import operator
import math


dfFull = pd.read_csv("breastCancer.csv")
classes = dfFull['class'].tolist()
columns = list(dfFull.columns)
features = columns[:len(columns)-1]
df = dfFull[features]


k = 2
MAX_ITER = 100
n = len(df)
m = 2.00

def initializeMembershipMatrix():
    membershipMat = list()
    for i in range(n):
        randomNumList = [random.random() for i in range(k)]
        summation = sum(randomNumList)
        tempList = [x/summation for x in randomNumList]
        membershipMat.append(tempList)
    return membershipMat


def calculateClusterCenter(membershipMat):
    clusterMemVal = list(zip(*membershipMat))
    clusterCenters = list()
    for j in range(k):
        x = list(clusterMemVal[j])
        xraised = [e ** m for e in x]
        denominator = sum(xraised)
        tempNum = list()
        for i in range(n):
            dataPoint = list(df.iloc[i])
            prod = [xraised[i] * val for val in dataPoint]
            tempNum.append(prod)
        numerator = list(map(sum, zip(*tempNum)))
        center = [z/denominator for z in numerator]
        clusterCenters.append(center)
    return clusterCenters


def updateMembershipValue(membershipMat, clusterCenters):
    p = float(2/(m-1))
    for i in range(n):
        x = list(df.iloc[i])
        distances = [np.linalg.norm(list(map(operator.sub, x, clusterCenters[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membershipMat[i][j] = float(1/den)       
    return membershipMat


def getClusters(membershipMat):
    clusterLabels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(membershipMat[i]))
        clusterLabels.append(idx)
    return clusterLabels


def fuzzyCMeansClustering():
    membershipMat = initializeMembershipMatrix()
    curr = 0
    while curr <= MAX_ITER:
        clusterCenters = calculateClusterCenter(membershipMat)
        membershipMat = updateMembershipValue(membershipMat, clusterCenters)
        clusterLabels = getClusters(membershipMat)
        curr += 1
    return clusterLabels, clusterCenters


labels, centers = fuzzyCMeansClustering()


tp = 0
tn = 0
fp = 0
fn = 0
cluster1 = 0
cluster2 = 0
count = 0
clusterss =[]
for i in labels:
    count = count + 1

    
    if i == 0:
        if classes[count-1] == 2:
            tn += 1
        else:
            fn +=1
        cluster1 += 1
        clusterss.append('2')
    else:
        if classes[count-1] == 4:
            tp += 1
        else:
            fp +=1
        cluster2 += 1
        clusterss.append('4')

#Main
sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+fn+tn+fp)
balancedAccuracy = (specificity+sensitivity)/2
total = 699
memberships = initializeMembershipMatrix()
membership = [memberships[0] for memberships in memberships]
np.savetxt("mem_degree.txt",np.array(membership), fmt = "%.2f")
np.savetxt("fuzzy_centers.txt",np.array(centers), fmt = "%.2f")
np.savetxt("fuzzy_clusters.txt",np.array(clusterss), fmt = "%s")
#for i in range(len(membership)):
    #label = labels[i]+1
    #print("Cluster:",label," Membership difference:",round(max(membership[i])-min(membership[i]), 2))
print("")
print("Number of data points in Cluster 1: " + str(cluster1))
print("Number of data points in Cluster 2: " + str(cluster2))
print("")
print("Specificity is: ", round(float(specificity),2))
print("Sensitivity is: ", round(float(sensitivity),2))
print("Balanced Accuracy is: ", round(float(balancedAccuracy),2))
print("Accuracy is: ", round(float(accuracy),2))






