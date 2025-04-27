import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]).reshape(4, 2, 1)

Y = np.array([
    0,
    1,
    1,
    0
])

weights_l0 = np.random.rand(2, 2)
bias_l1 = np.random.rand(2, 1)
weights_l1 = np.random.rand(1, 2)
bias_l2 = np.random.rand(1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feed_forward(x):
    z1 = np.dot(weights_l0, x) + bias_l1
    a1 = sigmoid(z1)
    
    z2 = np.dot(weights_l1, a1) + bias_l2
    y_hat = sigmoid(z2)

    return a1, y_hat


def L(y_hat, y):
    return 0.5 * (y_hat - y) ** 2


def sigmoid_derivative(x):
    return x * ( 1 - x )


def backprop(y_hat, y, w1, a1, xi):
    delta2 = (y_hat - y) * sigmoid_derivative(y_hat)
    dL_dw1, dL_db2 = delta2 @ a1.T, delta2
    delta1 = (w1.T @ delta2) * sigmoid_derivative(a1)
    dL_dw0, dL_db1 = delta1 @ xi.T, delta1
    return dL_dw0, dL_db1, dL_dw1, dL_db2


learning_rate = 1
epochs = 100000
losses = []


for i in tqdm(range(epochs)):
    loss = 0
    for i, x in enumerate(X):
        a1, y_hat = feed_forward(x)

        loss += L(y_hat, Y[i])

        dL_dw0, dL_db1, dL_dw1, dL_db2 = backprop(y_hat, Y[i], weights_l1, a1, x)
        weights_l0 -= learning_rate * dL_dw0
        weights_l1 -= learning_rate * dL_dw1
        bias_l1 -= learning_rate * dL_db1
        bias_l2 -= learning_rate * dL_db2

    losses.append(loss / 4)


xn = [i for i in range(epochs)]
plt.plot(xn, np.array(losses).reshape(epochs))
plt.show()

def classifier(prediction):
    if prediction >= 0.5: return 1
    else: return 0


for i in range(4):
    _, p = feed_forward(X[i]) 
    print(f'Input A: {X[i][0][0]}, Input B: {X[i][1][0]}, Prediction: {p[0][0]}')

example_num = 10000
point_A = np.random.rand(example_num).tolist()
point_B = np.random.rand(example_num).tolist()
coords = np.array(list(zip(point_A, point_B)))

pres = [0 for i in range(example_num)]
for i in range(example_num):
    _, p = feed_forward(coords[i].T.reshape(2, 1))
    pres[i] = classifier(p)

colors = ['red' if y == 0 else 'blue' for y in pres]


plt.scatter(point_A, point_B, color=colors)
plt.show()