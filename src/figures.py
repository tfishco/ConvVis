import matplotlib.pyplot as plt
import numpy as np

a = np.arange(-5,5,0.2)

def sig(a):
    return 1 / (1 + np.exp(-a))

def tanh(a):
    return (1 - np.exp(-2 * a)) / (1 + np.exp(-2 * a))

def relu(a,leaky):
    if leaky:
        return np.maximum(a,(0.1 *a))
    else:
        return np.maximum(a,0)

plt.figure(1)

plt.subplot(221)
plt.xlim(-5,5, 0.2)
plt.ylim(-0.5,1.5, 0.2)
plt.plot(a, sig(a), '-r')
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(222)
plt.xlim(-5,5, 0.2)
plt.ylim(-1.5,1.5, 0.2)
plt.plot(a, tanh(a), '-g')
plt.title('Tanh')
plt.grid(True)

#Leaky ReLU
plt.subplot(223)
plt.xlim(-5,5, 0.2)
plt.ylim(-1,5, 0.2)
plt.plot(a, relu(a, leaky=False), '-b')
plt.title('ReLU')
plt.grid(True)

#ReLU
plt.subplot(224)
plt.xlim(-5,5, 0.2)
plt.ylim(-1,5, 0.2)
plt.plot(a, relu(a, leaky=True), 'k')
plt.title('Leaky Relu')
plt.grid(True)

plt.show()
