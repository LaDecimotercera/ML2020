import pandas as pd
import matplotlib.pyplot as plt

def plot_PR(x_A, y_A):
    plt.plot(x_A, y_A, linewidth=2.5, linestyle='-', label='A')
    plt.title('PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('PR_curve.png')
    plt.clf()


def plot_ROC(x_A, y_A):
    plt.plot(x_A, y_A, linewidth=2.5, linestyle='-', label='A')
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig('ROC_curve.png')
    plt.clf()


def AUC(x_A, y_A):
    area = 0
    for i in range(len(x_A)-1):
        area += 0.5 * (x_A[i+1] - x_A[i]) * (y_A[i+1] + y_A[i])
    return area


if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    df = df.sort_values(by='output')
    output = df['output'].values
    label = df['label'].values
    N = list(label).count(0)
    P = list(label).count(1)
    TP = 0
    FP = 0
    FN = P
    TN = N
    PR_list = [(0, 1)]
    ROC_list = [(0, 0)]
    last = -1
    output = list(output)
    output.reverse()
    for i, y in enumerate(reversed(label)):
        if y == 1:
            TP += 1.
            FN -= 1.
        else:
            FP += 1.
            TN -= 1.
        if i < len(output) - 1 and output[i+1] == output[i]:
            continue
        precision = TP/(TP + FP)
        recall = TP/(TP+FN)
        TPR = TP/(TP+FN)
        FPR = FP/(TN+FP)
        PR_list.append((recall, precision))
        ROC_list.append((FPR, TPR))
    x_PR = [x for x, _ in PR_list]
    y_PR = [y for _, y in PR_list]
    x_ROC = [x for x, _ in ROC_list]
    y_ROC = [y for _, y in ROC_list]
    plot_PR(x_PR, y_PR)
    plot_ROC(x_ROC, y_ROC)
    print(AUC(x_ROC, y_ROC))