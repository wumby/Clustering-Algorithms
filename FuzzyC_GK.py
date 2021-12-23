#Jack Ziegler Gustafson Kessel
#Breast Cancer dataset
import numpy as np
import pandas as pd
import random


class fuzzyC:
	def __init__(self):
		self.c = 2
		self.N = 0
		self.n = 0
		self.m = None
		self.epsilon = None
		self.A = None
		self.V = None
		self.U = None

	def set(self,
			Z,
			c,
			fuzzyParameter=2,
			terminationCriterion=0.01,
			normalMatrix='identity'):
		self.__init_vars(Z, c, fuzzyParameter, terminationCriterion, normalMatrix)
		lastU = 0
		firstTime = True
		while firstTime or not self.reachedTermination(lastU):
			firstTime = False
			self.calcClusterMeans(Z)
			D = self.__compute_distances(Z)

			lastU = np.zeros([self.c, self.N])
			for i in range(self.c):
				for k in range(self.N):
					lastU[i][k] = self.U[i][k]

			self.updateMatrix(D, Z)
		

	def predict(self, Z):
		return self.calcPartitionMatrix(Z)

	def __init_vars(self, Z, c, fuzzyParameter, terminationCriterion, normalMatrix):
		self.c = c
		self.N = Z.shape[1]
		self.n = Z.shape[0]
		self.m = fuzzyParameter
		self.epsilon = terminationCriterion
		self.__init_A(normalMatrix, Z)
		self.__init_V(Z)
		self.__init_U(Z)

	def __init_A(self, normalMatrix, Z):
		if normalMatrix == 'identity':
			self.A = np.eye(self.n)
		elif normalMatrix == 'diagonal':
			z_var = np.zeros([self.n, 1])

			for i in range(self.n):
				tmp = Z[i, :]
				z_var[i] = 1/np.var(tmp)

			self.A = np.diagflat(np.reshape(z_var, [self.n, 1]))
			return
		elif normalMatrix == 'mahalonobis':
			zMean = np.zeros([self.n, 1])

			for i in range(self.n):
				tmp = Z[i, :]
				zMean[i] = np.mean(tmp)

			R = np.zeros([self.n, self.n])
			for i in range(self.N):
				zk = np.reshape(Z[:, i], [self.n, 1])
				diff = np.subtract(zk, zMean)
				R = np.add(R, np.matmul(diff, diff.transpose()))

			R = 1/self.N * R
			self.A = np.linalg.inv(R)

	def __init_V(self, Z):
		self.V = np.zeros([self.n, self.c])
		for cluster in range(self.c):
			for feature in range(self.n):
				self.V[feature][cluster] = random.uniform(np.min(Z[feature]), np.max(Z[feature]))

	def __init_U(self, Z):
		self.U = np.zeros([self.c, self.N])

		D = self.__compute_distances(Z)
		self.updateMatrix(D, Z)

	def __D_squared(self, i, k, Z, V):
		zk = np.reshape(Z[:, k], [Z.shape[0], 1])
		vi = np.reshape(V[:, i], [V.shape[0], 1])

		diff = np.subtract(zk, vi)
		ret = np.matmul(diff.transpose(), self.A)
		ret = np.matmul(ret, diff)

		return ret

	def reachedTermination(self, lastU):
		diff = np.max(np.abs(np.subtract(self.U, lastU)))
		if diff < self.epsilon:
			return True
		return False

	def calcClusterMeans(self, Z):
		for i in range(self.c):
			nom = np.zeros([self.n, 1])
			denom = 0

			for k in range(self.N):
				muPower = pow(self.U[i][k], self.m)
				denom += muPower
				nom = np.add(nom, muPower * np.reshape(Z[:, k], [Z.shape[0], 1]))

			for row in range(self.V.shape[0]):
				self.V[row][i] = nom[row]/denom

	def __compute_distances(self, Z):
		D = np.zeros([self.c, self.N])

		for i in range(self.c):
			for k in range(self.N):
				D[i][k] = np.sqrt(self.__D_squared(i, k, Z, self.V))

		return D

	def updateMatrix(self, D, Z):
		for k in range(self.N):
			all_distances_positive = True
			for i in range(self.c):
				if D[i][k] == 0:
					all_distances_positive = False

			if all_distances_positive:
				for i in range(self.c):
					d_ik = np.sqrt(self.__D_squared(i, k, Z, self.V))
					sum_distance = 0

					for j in range(self.c):
						d_jk = np.sqrt(self.__D_squared(j, k, Z, self.V))
						add = pow(d_ik / d_jk, 2 / (self.m - 1))
						sum_distance += add[0]

					self.U[i][k] = 1/sum_distance
			else:
				edit = []
				for i in range(self.c):
					if D[i][k] > 0:
						self.U[i][k] = 0
					else:
						edit.append(i)

				remaining = 1
				sum_added = 0
				for i in range(len(edit) - 1):
					self.U[edit[i]][k] = random.uniform(0, remaining)
					sum_added += self.U[edit[i]][k]
					remaining -= self.U[edit[i]][k]

				self.U[edit[len(edit) - 1]][k] = remaining

	def calcPartitionMatrix(self, Z):
		U = np.zeros([self.c, Z.shape[1]])

		for k in range(Z.shape[1]):
			for i in range(self.c):
				d_ik = np.sqrt(self.__D_squared(i, k, Z, self.V))
				sum_distance = 0

				for j in range(self.c):
					d_jk = np.sqrt(self.__D_squared(j, k, Z, self.V))
					add = pow(d_ik / d_jk, 2 / (self.m - 1))
					sum_distance += add[0]

				U[i][k] = 1 / sum_distance
		return U

class GK(fuzzyC):
	def __init__(self):
		self.F = []
		self.rho = None
		super(GK, self).__init__()

	def __init_vars(self, Z, c, fuzzyParameter, terminationCriterion, normalMatrix):
		self.rho = np.ones(c)
		super(GK, self).__init_vars(Z, c, fuzzyParameter, terminationCriterion, normalMatrix)

	def __init_U(self, Z):
		self.U = np.zeros([self.c, self.N])

		for k in range(self.N):
			remaining = 1
			sum_added = 0
			for i in range(self.c - 1):
				self.U[i][k] = random.uniform(0, remaining)
				sum_added += self.U[i][k]
				remaining -= self.U[i][k]

			self.U[self.c - 1][k] = remaining
		assert abs(np.sum(self.U) - self.N) < self.epsilon, 'U initialized incorrectly'

	def __compute_distances(self, Z):
		self.calcCovariance(Z)
		return super(GK, self).__compute_distances(Z)

	def __D_squared(self, i, k, Z, V):
		zk = np.reshape(Z[:, k], [Z.shape[0], 1])
		vi = np.reshape(V[:, i], [V.shape[0], 1])

		A = pow(self.rho[i] * np.linalg.det(self.F[i]), 1/self.n) * np.linalg.inv(self.F[i])

		diff = np.subtract(zk, vi)
		ret = np.matmul(diff.transpose(), A)
		ret = np.matmul(ret, diff)

		return ret

	def calcCovariance(self, Z):
		for i in range(self.c):
			muPower = 0
			vi = np.reshape(self.V[:, i], [self.V.shape[0], 1])

			f = np.zeros([self.n, self.n])
			for k in range(self.N):
				muPower += pow(self.U[i][k], self.m)

				zk = np.reshape(Z[:, k], [Z.shape[0], 1])
				diff = np.subtract(zk, vi)
				add = pow(self.U[i][k], self.m) * np.matmul(diff, diff.transpose())
				f = np.add(f, add)

			self.F.append(f/muPower)

def calculate(model, data):
	
	Z = data.transpose()
	clusters = []
	for k in range(Z.shape[1]):
		sample = np.reshape(Z[:, k], [Z.shape[0], 1])
		pred = np.argmax(model.predict(sample)) + 1
		clusters.append(pred)
	return clusters
		

#Main
tp = 0
tn = 0
fp = 0
fn = 0
datas = pd.read_csv("breastCancer.csv")
datas.pop('class')
data = datas.iloc[:,:-1].values
Z = data.transpose()
gk = GK()
gk.set(Z, 2)
clusters = calculate(gk, data)
read = pd.read_csv("breastCancer.csv")
classes = read['class'].tolist()

count = 0
for i in clusters:
	count = count +1
	if i == 1:
		if classes[count-1] == 2:
			tn+=1
		else:
			fn+=1
	else:
		if classes[count-1] == 4:
			tp+=1
		else:
			fp+=1

sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)
accuracy = (tp+tn)/(tp+fn+tn+fp)
balancedAccuracy = (specificity+sensitivity)/2       
print("Specificity is: ", round(float(specificity),2))
print("Sensitivity is: ", round(float(sensitivity),2))
print("Balanced Accuracy is: ", round(float(balancedAccuracy),2))
print("Accuracy is: ", round(float(accuracy),2))

print(clusters)
np.savetxt("GK_classes.txt",np.array(classes), fmt = "%s")