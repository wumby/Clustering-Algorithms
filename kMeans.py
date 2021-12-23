import pandas as pd
from random import sample
from math import sqrt
from numpy import mean
import numpy as np
def newCenters(df, k):
    randomIndices = sample(range(len(df)), k)
    centers = [list(df.iloc[idx]) for idx in randomIndices]
    return centers

def calcCenter(df, k, clusterLabels):
    clusterCenters = list()
    dataPoints = list()
    for i in range(k):
        for idx, val in enumerate(clusterLabels):
            if val == i:
                dataPoints.append(list(df.iloc[idx]))
        clusterCenters.append(list(map(mean, zip(*dataPoints))))
    return clusterCenters

def euclideanDistance(x, y):
    summ = 0
    for i in range(len(x)):
        term = (x[i] - y[i])**2
        summ += term
    return sqrt(summ)

def assignCluster(df, k, clusterCenters):
    clusterAssigned = list()
    for i in range(len(df)):
        distances = [euclideanDistance(list(df.iloc[i]), center) for center in clusterCenters]
        min_dist, idx = min((val, idx) for (idx, val) in enumerate(distances))
        clusterAssigned.append(idx)
    return clusterAssigned


def kmeans(df, k, class_labels):
    clusterCenters = newCenters(df, k)
    curr = 1
    
    while curr <= MAX_ITER:
        clusterLabels = assignCluster(df, k, clusterCenters)
        clusterCenters = calcCenter(df, k, clusterLabels)
        curr += 1   
    return clusterLabels, clusterCenters


k = 2
MAX_ITER = 100 
df_full = pd.read_csv('breastCancer.csv')
columns = list(df_full.columns)
features = columns[:len(columns)-1]
class_labels = list(df_full[columns[-1]])
df = df_full[features]


labels, centers = kmeans(df, k, class_labels)
cluster1 = 0
cluster2 = 0
count = 0
tp = 0
tn = 0
fp =0
fn = 0
for i in labels:
    count = count + 1
    
    if i == 0:
        if class_labels[count-1] == 2:
            string = 'True'
            tn += 1
        else:
            string = 'False'
            fn += 1
        print("Row",count, "is in cluster 2:", class_labels[count-1], " is the actual class:", string)
        cluster1 += 1
    else:
        if class_labels[count-1] == 4:
            string = 'True'
            tp += 1
        else:
            string = 'False'
            fp += 1
        print("Row",count, "is in cluster 4:", class_labels[count-1], " is the actual class", string)
        cluster2 += 1

sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+fn+tn+fp)
balancedAccuracy = (specificity+sensitivity)/2
print("Number of data points in Cluster 1: " + str(cluster1))
print("Number of data points in Cluster 2: " + str(cluster2))
print("")
print("Specificity is: ", round(float(specificity),2))
print("Sensitivity is: ", round(float(sensitivity),2))
print("Balanced Accuracy is: ", round(float(balancedAccuracy),2))
print("Accuracy is: ", round(float(accuracy),2))
np.savetxt("kmeans_clusters.txt",np.array(class_labels), fmt = "%s")
print(centers)