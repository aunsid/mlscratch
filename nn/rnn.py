import numpy as np
import matplotlib.pyplot as plt


class Tanh():
    def forward(self, inputs):
        self.output = np.tanh(inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        out = 1 - (self.output ** 2)
        out = out * dvalues
        self.dinputs = out
        return out

class RNN():
    def __init__(self, X_t, n_neurons, activation):
        self.T = max(X_t.shape) # time points
        self.X_t = X_t
        self.Y_hat = np.zeros((self.T, 1)) # many to many
        self.activation = activation

        self.n_neurons = n_neurons

        # self.h = 0.1 * np.random.randn()

        self.Wx = 0.1 * np.random.randn(n_neurons, 1)
        self.Wh = 0.1 * np.random.randn(n_neurons, n_neurons)
        self.Wy = 0.1 * np.random.randn(1, n_neurons)
        self.bias = 0.1 * np.random.randn(n_neurons, 1)

        # states 
        self.H = [np.zeros((n_neurons, 1)) for t in range(self.T+1)]

    def forward(self):
        self.dWx = np.zeros_like(self.Wx)
        self.dWh = np.zeros_like(self.Wh)
        self.dWy = np.zeros_like(self.Wy)
        self.dbias = np.zeros_like(self.bias)

        X_t = self.X_t
        H = self.H
        Y_hat = self.Y_hat
        ht = H[0]
        ACT = [self.activation.__class__() for _ in range(self.T)]

        ACT, H, Y_hat = self.rnn_cell(X_t, ht, ACT, H, Y_hat)
        self.ACT = ACT

    def rnn_cell(self, X_t, ht, ACT, H, Y_hat):
        
        for t, xt in enumerate(X_t):
            
            xt = xt.reshape(1, 1)
            out     = self.Wx @ xt + self.Wh @ ht + self.bias
            ACT[t].forward(out)
            ht = ACT[t].output
            y_hat_t = self.Wy @ ht

            H[t+1] = ht
            Y_hat[t] = y_hat_t

        return ACT, H, Y_hat
    
    def backward(self, dvalues):

        X_t = self.X_t
        H = self.H
        Y_hat = self.Y_hat
        ht = H[0]
        ACT = self.ACT
       

        dWx = self.dWx
        dWh = self.dWh
        dWy = self.dWy
        dbias = self.dbias
        
        Wh = self.Wh
        Wy = self.Wy
        dht = Wy.T @ dvalues[-1].reshape(1, 1)

        for t in reversed(range(self.T)):
            dy = dvalues[t].reshape(1, 1)
            xt = X_t[t].reshape(1, 1)

            ACT[t].backward(dht)
            dtanh = ACT[t].dinputs

            dWx += dtanh @ xt.T
            dWy += dy @ H[t+1].T
            dWh += dtanh @ H[t].T
            dbias += dtanh

            dht = Wh @ dtanh + Wy.T @ dy

        self.dWx = dWx / self.T
        self.dWy = dWy / self.T
        self.dWh = dWh / self.T
        self.dbias = dbias / self.T

    
    # def cell(self, xt, ht_1):
    #     out     = self.Wx @ xt + self.Wh @ ht_1 + self.bias
    #     ht      = np.tanh(out)
    #     y_hat_t = self.Wy @ ht
    #     return ht, y_hat_t, out

    


if __name__ == "__main__":

    n_neurons = 500
    n_epoch = 200000
    e = 1e-4

    x = np.arange(-10, 10, 0.1)
    X_t = x.reshape(x.shape[0], 1)
    Y_t = np.sin(X_t) + 0.1 * np.random.randn(len(x), 1)

    rnn = RNN(X_t, n_neurons, Tanh())
    T = rnn.T
   

    for n in range(n_epoch):
        rnn.forward()
        Y_hat = rnn.Y_hat
        dY = Y_hat - Y_t
        L = 0.5 * np.dot(dY.T, dY)/T
        print(float(L))
        rnn.backward(dY)

        rnn.Wx  -= e * rnn.dWx
        rnn.Wy  -= e * rnn.dWy
        rnn.Wh  -= e * rnn.dWh
        rnn.bias -= e * rnn.dbias

    # Y_hat = rnn.Y_hat
    # H = rnn.H
    # T = rnn.T

    # dY = Y_hat - Y_t
    # L = 0.5 * np.dot(dY.T, dY)/ T
  
    # rnn = RNN(X_t, n_neurons)

    # Y_hat = rnn.Y_hat
    # H = rnn.H
    # T = rnn.T
    # ht = H[0]

    # for t, xt in enumerate(X_t):
    #     xt = xt.reshape(1, 1)
    #     [ht, y_hat_t, out] = rnn.forward(xt, ht)
    #     H[t+1] = ht
    #     Y_hat[t] = y_hat_t 


    plt.figure()
    plt.plot(X_t, Y_t)
    plt.plot(X_t, Y_hat)
    plt.savefig('rnn_plot.png')
    plt.show()







