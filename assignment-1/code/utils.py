import numpy as np
import scipy.stats as stats
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, pairwise_distances 


def load_data():
    data = loadmat('digits.mat')
    X = data['X']
    Y = data['Y']

    return X, Y


class ClassConditionalsAndPriors:
    def __init__(self, mat, labels, class_):
        self.data = mat[(labels == class_).reshape(len(labels))]
        self.class_ = class_

        self.m = self.data.shape[0]
        self.d = self.data.shape[1]
    
        # class-conditionals
        self.μ = None
        self.Σ = None
        self.multivarNormal = None
        
        # class-priors
        self.ρ = self.m/mat.shape[0]
        
    def mle_estimates(self):
        self.μ = (1/self.m) * np.sum(self.data, axis=0).reshape(-1,1)
        
        self.Σ = np.zeros((self.d, self.d))
        for i in range(self.m):
            self.Σ += (self.data[i].reshape(-1,1) - self.μ) @ (self.data[i].reshape(-1,1) - self.μ).T
        
        self.Σ *= 1/self.m
        self.Σ += np.identity(self.d) * .1
        self.multivarNormal = stats.multivariate_normal(mean=self.μ.reshape(-1,), cov=self.Σ)


class MLEClassifier:
    def __init__(self):
        self.L = None
        self.classes = None
        
        self.ccnp = None  

    def fit(self, data, labels):
        self.classes = np.unique(labels).tolist()
        self.L = len(self.classes)

        cc = [None for _ in range(self.L)]

        for c in self.classes:
            cc[c] = ClassConditionalsAndPriors(data, labels, c)
            cc[c].mle_estimates()

        self.ccnp = cc
    
    def predict(self, data, return_probs=False):
        pred = []
        probs = []
        for i in range(data.shape[0]):
            maxProb = np.NINF

            for c in self.classes:
                currProb = self.ccnp[c].multivarNormal.logpdf(x=data[i])
                currProb *= self.ccnp[c].ρ

                if currProb > maxProb:
                    maxProbLabel = c
                    maxProb = currProb

            pred.append(maxProbLabel)
            probs.append(maxProb)
        
        pred = np.array(pred).reshape(-1,)

        if return_probs == True:
            return pred, probs

        return pred

    def score(self, data, true):
        return accuracy_score(y_pred=self.predict(data), y_true=true.reshape(-1,))


class KNNClassifier:
    def __init__(self):
        self.m = None
        self.d = None
        self.data = None
        self.labels = None

    def fit(self, data, labels):
        self.m = data.shape[0]
        self.d = data.shape[1]
        self.data = data
        self.labels = labels

    def predict(self, pMat, K, p):
        if pMat.ndim == 1:
            pMat = pMat.reshape(1, -1)
        
        pred = []
        if p == np.inf:
            distances = pairwise_distances(X=pMat, Y=self.data, metric='chebyshev')
        elif p == 1:
            distances = pairwise_distances(X=pMat, Y=self.data, metric='manhattan')
        else:
            distances = pairwise_distances(X=pMat, Y=self.data)

        nearestNeighbors = np.argsort(distances, axis=1)[:, 0:K]
        nearestLabels = self.labels.reshape(-1,)[nearestNeighbors]
        pred = (stats.mode(nearestLabels, axis=1))[0].reshape(-1,)

        return pred
        
    def score(self, pMat, true, K, p):
        pred = self.predict(pMat, K, p)
        return accuracy_score(y_true=true.reshape(-1,), y_pred=pred.reshape(-1,))

