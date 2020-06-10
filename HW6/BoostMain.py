from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

class AdaBoost:

    def __init__(self, n_estimators):
        self.base = []
        self.n_estimators = n_estimators
        self.alpha = np.zeros(n_estimators)
        
    def fit(self, X, y):
        length = X.shape[0]
        D = np.ones(length) / length
        # training
        for t in range(self.n_estimators):
            clf = DecisionTreeClassifier()#criterion="entropy", splitter="random", ccp_alpha=0.001)
            clf = clf.fit(X, y, sample_weight = D)
            y_pred = clf.predict(X)
            error = 1 - clf.score(X, y, sample_weight = D)
            
            if error > 0.5:
                break
            elif error == 0:
                self.base.append(clf)
                self.alpha[t] = 1
                break
                
            self.base.append(clf)
            self.alpha[t] = 0.5 * np.log((1 - error) / error)
            D = D * np.exp(-self.alpha[t]*y_pred*y)
            D = D / np.sum(D)
    
    def predict(self, X):
        H = np.zeros(X.shape[0])
        for t in range(len(self.base)):
                H += self.alpha[t]*self.base[t].predict(X)
            
        H[H>=0] = 1
        H[H<0] = -1
        return H
    
    def predict_proba(self, X):
        prob = np.zeros(X.shape[0])
        for t in range(len(self.base)):
            prob += (self.alpha[t]/self.alpha.sum())*self.base[t].predict_proba(X)[:,1]
            
        return prob
    
    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

def crossValidation(X, y):
    kf = StratifiedKFold(n_splits=5)
    x_axis =[]
    y_axis = []
    for t in range(1, 51):# test n_estimators
        AUC = 0
        x_axis.append(t)
        for train_index, val_index in kf.split(X,y):
            train_data, val_data = X[train_index], X[val_index]
            train_label, val_label = y[train_index], y[val_index]
            clf = AdaBoost(n_estimators=t)
            clf.fit(train_data,train_label)
            pred = clf.predict(val_data)
            prob = clf.predict_proba(val_data)
            AUC += roc_auc_score(val_label, prob)
        AUC /= 5
        y_axis.append(AUC)
        print("NUM = ",t," Valiadtion AUC = ", AUC)
    return x_axis, y_axis
    
if __name__ == "__main__":
    # load data from source
    X_train = np.genfromtxt("adult_dataset/adult_train_feature.txt")
    X_test  = np.genfromtxt("adult_dataset/adult_test_feature.txt")
    y_train = np.genfromtxt("adult_dataset/adult_train_label.txt")
    y_test  = np.genfromtxt("adult_dataset/adult_test_label.txt")
    # preprocess
    y_train = (y_train - 0.5) * 2
    y_test = (y_test - 0.5) * 2    
    # train & predict
    clf = AdaBoost(n_estimators = 50)
    clf.fit(X_train, y_train)
    print("(Adaboost) accuracy on testing set: ",clf.score(X_test, y_test))
    
    # cross-validation
    #x1,y1 = crossValidation(X_train,y_train)
    
    # auc for optimal n_estimator
    #pred = clf.predict(X_test)
    #prob = clf.predict_proba(X_test)
    #res = roc_auc_score(y_test, prob)
    #print("NUM=30, AUC=", res)
    
    # plot
    #plt.plot(x1,y1,',-',label='AdaBoost')
    #plt.xlabel("num of learners")
    #plt.ylabel("Validation AUC")
    #plt.legend()
    #plt.show()
