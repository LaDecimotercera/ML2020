#coding:utf-8
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./data.csv')
#Preprocessing
dataset.sort_values("output",ascending=False,inplace=True)
output = dataset["output"].values.tolist()
recall = []
precision = []
TP_Rate = []
FP_Rate = []

def gen_params(threshold):
    TN = dataset[(dataset["output"]<threshold) & (dataset["label"]==0)]["Index"].count()
    FP =dataset[(dataset["output"]>=threshold) & (dataset["label"]==0)]["Index"].count()
    FN =dataset[(dataset["output"]<threshold) & (dataset["label"]==1)]["Index"].count()
    TP = dataset[(dataset["output"]>=threshold) & (dataset["label"]==1)]["Index"].count()
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    precision.append(P)
    recall.append(R)
    TPR = TP / (TP + FN)
    FPR = FP / (TN + FP)
    TP_Rate.append(TPR)
    FP_Rate.append(FPR)

for th in output:
    gen_params(th)
# P-R Graph
plt.figure()
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot(recall, precision)
plt.title("P-R Curve")
plt.show()

# ROC Graph
plt.figure()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.plot(FP_Rate,TP_Rate)
plt.title("ROC Curve")
plt.show()

AUC = 0
for idx in range(len(output)-1):
    AUC += (FP_Rate[idx+1]-FP_Rate[idx])*(TP_Rate[idx+1]+TP_Rate[idx])/2
print(AUC)    