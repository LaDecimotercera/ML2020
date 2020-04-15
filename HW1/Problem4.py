import numpy as np
import math

sample = np.array([[2,3,1,5,4],[5,4,2,3,1],[4,5,1,2,3],[2,3,1,5,4],[3,4,1,5,2]])
ri = sample.mean(axis=0)
sum = 0
for i in range(len(ri)):
    sum += ri[i]**2

#Friedman a=0.05 
#Fa = 3.007
#Nemenyi qa=2.728
def calcT_X2(N, k):
    return (sum - k*(k+1)*(k+1)/4)*(12*N/(k*(k+1)))

def calT_F(N, k):
    T_X2 = calcT_X2(N,k)
    return (N-1)*T_X2/(N*(k-1)-T_X2)

print(calT_F(5,5))#不符合假设 cuz > Fa

def calcCD(N, k, qa):
    return qa*math.sqrt(k*(k+1)/(6*N))

print(calcCD(5,5,2.728))