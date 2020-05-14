import numpy as np

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# loss
def mse_loss(y_true, y_pred):
    return np.square(np.subtract(y_true,y_pred)).sum()/2

def cross_entropy(targets, predictions):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce

def accuracy_score(targets, predictions):
    return np.mean(targets==predictions)

class NeuralNetwork_221():
    def __init__(self):
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        # 以上为神经网络中的变量，其中具体含义见网络图

    def predict(self,x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 500
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # 以下部分为向前传播过程，请完成
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2) 
                
                sum_ol = self.w5 * h1 + self.w6 * h2 + self.b3 
                ol = sigmoid(sum_ol) 
                y_pred = ol

                # 以下部分为计算梯度，请完成
                # For Cross Entrophy
                d_L_d_ypred = - y_true / y_pred + (1 - y_true) / (1 - y_pred) 
                # For MSE: y_pred - y_true
                
                # （需要填写的地方，含义为损失函数对输出层对率输出的梯度）
                # 输出层梯度
                d_ypred_d_w5 = deriv_sigmoid(sum_ol) * h1 # （需要填写的地方，含义为输出层对率输出对w5的梯度）
                d_ypred_d_w6 = deriv_sigmoid(sum_ol) * h2 # （需要填写的地方，含义为输出层对率输出对w6的梯度）
                d_ypred_d_b3 = deriv_sigmoid(sum_ol) # （需要填写的地方，含义为输出层对率输出对b3的梯度）
                d_ypred_d_h1 = deriv_sigmoid(sum_ol) * self.w5 # （需要填写的地方，含义为输出层输出对率对隐层第一个节点的输出的梯度）
                d_ypred_d_h2 = deriv_sigmoid(sum_ol) * self.w6 # （需要填写的地方，含义为输出层输出对率对隐层第二个节点的输出的梯度）

                # 隐层梯度
                d_h1_d_w1 = deriv_sigmoid(sum_h1) * x[0] # （需要填写的地方，含义为隐层第一个节点的输出对w1的梯度）
                d_h1_d_w2 = deriv_sigmoid(sum_h1) * x[1] # （需要填写的地方，含义为隐层第一个节点的输出对w2的梯度）
                d_h1_d_b1 = deriv_sigmoid(sum_h1) # （需要填写的地方，含义为隐层第一个节点的输出对b1的梯度）

                d_h2_d_w3 = deriv_sigmoid(sum_h2) * x[0] # （需要填写的地方，含义为隐层第二个节点的输出对w3的梯度）
                d_h2_d_w4 = deriv_sigmoid(sum_h2) * x[1] # （需要填写的地方，含义为隐层第二个节点的输出对w4的梯度）
                d_h2_d_b2 = deriv_sigmoid(sum_h2) # （需要填写的地方，含义为隐层第二个节点的输出对b2的梯度）

                # 更新权重和偏置
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5 # （需要填写的地方，更新w5）
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6 # （需要填写的地方，更新w6）
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3 # （需要填写的地方，更新b3）

                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1 # （需要填写的地方，更新w1）
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2 # （需要填写的地方，更新w2）
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1 # （需要填写的地方，更新b1）

                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3 # （需要填写的地方，更新w3）
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4 # （需要填写的地方，更新w4）
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2 # （需要填写的地方，更新b2）

            # 计算epoch的loss
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.predict, 1, data)
                loss = cross_entropy(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" %(epoch, loss))
                
def main():
    import numpy as np
    X_train = np.genfromtxt('./data/train_feature.csv', delimiter=',')
    y_train = np.genfromtxt('./data/train_target.csv', delimiter=',')
    X_test = np.genfromtxt('./data/test_feature.csv', delimiter=',')#读取测试样本特征
    network = NeuralNetwork_221()
    network.train(X_train, y_train)
    y_pred=[]
    theta = 0.5
    for i in X_train:
        #y_pred.append(network.predict(i))#将预测值存入y_pred(list)内
        label = 1 if (network.predict(i) > theta) else 0
        y_pred.append(label)
    #print(accuracy_score(np.array(y_pred), y_train))
    np.savetxt("./181220031_ypred.csv", np.asarray(y_pred), fmt="%d", delimiter=',')
    
main()
