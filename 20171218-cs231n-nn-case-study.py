# ==========================================================================
# CS231n Neural Net Case Study
# --------------------------------------------------------------------------
# Code Reference: http://cs231n.github.io/neural-networks-case-study/#grad
# 
# Practice writing Neural Net using only NumPy.
# ==========================================================================

import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2
K = 3
X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='uint8')
for j in range(K):
    ix = range(N*j, N*(j+1)) #+ how to generate spiral dataset
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

#plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

# =========================
# Softmax Linear Classifier
# =========================

num_iter = 100
display_step = 20
learning_rate = 2e-0
reg = 1e-3
num_data = X.shape[0]

W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

for i in range(num_iter):
    # Forward
    scores = np.dot(X, W) + b # [N,K]
    sftmx = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) # [N,K]
    logprob = -np.log(sftmx[range(num_data), y]) # [N,1]
    data_loss = np.sum(logprob)/num_data
    reg_loss = 0.5*reg*np.sum(W**2)
    loss = data_loss + reg_loss

    if i % display_step == 0:
        print('iter {}: loss {}'.format(i, loss))

    # Back propogate
    dscores = sftmx
    dscores[range(num_data), y] -= 1 # shape = num*K
    dscores /= num_data #+ divide the gradient by num ptrs
    dW = np.dot(X.T, dscores) # shape = D*K
    db = np.sum(dscores, axis=0, keepdims=True)
    dW += reg*W # [D,K]

    # Update
    W -= learning_rate * dW
    b -= learning_rate * db

scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print("Training accuracy {}".format(np.mean(predicted_class==y)))


# =========================
# Neural Network
# =========================

h = 100
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

learning_rate = 1e-0
reg = 1e-3
num_iter = 10000
num_data = X.shape[0]
display_step = 1000

for i in range(num_iter):
    hidden_layer = np.maximum(0, np.dot(X, W)+b) # [N*h]
    scores = np.dot(hidden_layer, W2) + b2 # [N*K]
    sftmx = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True) # [N,K]
    logprob = -np.log(sftmx[range(num_data), y]) #[N,1]
    data_loss = np.sum(logprob) / num_data
    reg_loss = 0.5*reg*np.sum(W**2) + 0.5*reg*np.sum(W2**2)
    loss = data_loss + reg_loss

    if i % display_step == 0:
        print('iter {}: loss {}'.format(i, loss))

    dscores = sftmx # [N*K]
    dscores[range(num_data), y] -= 1
    dscores /= num_data
    dW2 = np.dot(hidden_layer.T, dscores) # [h*K]
    db2 = np.sum(dscores, axis=0, keepdims=True) #[1,K]
    dhidden = np.dot(dscores, W2.T) # [N*h]
    dhidden[hidden_layer <= 0] = 0 #+ How to back propagate through ReLU
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True) #[1,h]

    dW2 += reg*W2
    dW += reg*W

    W -= learning_rate*dW
    b -= learning_rate*db
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2

hidden_layer = np.maximum(0, np.dot(X, W)+b) 
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print("Training accuracy {}".format(np.mean(predicted_class==y)))
