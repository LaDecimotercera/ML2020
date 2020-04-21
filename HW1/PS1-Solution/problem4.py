import pandas as pd
import numpy as np

average_performace = [3.2, 3.8, 1.2, 4, 2.8]
df = pd.DataFrame({'A': [2, 5, 4, 2, 3], 'B': [3, 4, 5, 3, 4], 'C': [1, 2, 1, 1, 1], 'D':
                   [5, 3, 2, 5, 5], 'E': [4, 1, 3, 4, 2]})


def f(df: pd.DataFrame):
    k = len(df.columns)
    N = len(df.index)
    S = 0
    for i in range(k):
        performance = df.iloc[:, i].values
        mean_performance = np.mean(performance)
        S += mean_performance ** 2
    forigin = 12 * N / (k * (k+1)) * (S - k * ((k+1)**2)/4)
    F = (N-1) * forigin / (N * (k-1) - forigin)
    return F


if __name__ == '__main__':
    tau = f(df)
    print(tau)
    CD = 2.728

