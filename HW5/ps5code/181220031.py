from __future__ import print_function
import numpy as np
import pandas as pd

class KNN():
   
    #k: int,最近邻个数.
    def __init__(self, k=5):
        self.k = k
        
    def distance(self, one_sample, X_train):
        dist = [np.linalg.norm(one_sample - x) for x in X_train]
        return dist
    
    # 此处需要填写，获取k个近邻的类别标签
    def get_k_neighbor_labels(self, distances, y_train, k):
        idx = np.argsort(distances)[:k]
        k_nearest_neighbors = np.array([y_train[i] for i in idx])
        return k_nearest_neighbors
    
    # 此处需要填写，标签统计，票数最多的标签即该测试样本的预测标签
    def vote(self, one_sample, X_train, y_train, k):
        dist = self.distance(one_sample, X_train)
        neighbor_labels = self.get_k_neighbor_labels(dist, y_train, k)
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()
    
    # 此处需要填写，对测试集进行预测
    def predict(self, X_test, X_train, y_train):
        y_preds = np.apply_along_axis(self.vote, 1, X_test, X_train, y_train, self.k)
        return y_preds
  

def main():
    clf = KNN(k=5)
    train_data = np.genfromtxt('./data/train_data.csv', delimiter=' ')
    train_labels = np.genfromtxt('./data/train_labels.csv', delimiter=' ')
    test_data = np.genfromtxt('./data/test_data.csv', delimiter=' ')
   
    #将预测值存入y_pred(list)内    
    y_pred = clf.predict(test_data, train_data, train_labels)
    np.savetxt("test_ypred.csv", y_pred, fmt="%d", delimiter=' ')


if __name__ == "__main__":
    main()