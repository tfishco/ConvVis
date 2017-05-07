import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

dataset = 'mnist'
iterations = 1000
batches = 50

x_data = []
y_data = []

with open("cifar-1000-50.txt", "r") as f:
    for line in f:
        line_split = line.split(":")
        y_data.append([float(line_split[1]),float(line_split[2]),float(line_split[3].strip())])


with open("cifar-1000-50-dropout.txt", "r") as f:
    for line in f:
        x_data.append(int(line.split(":")[0]))

def average_axis(data):
    avg = []
    for i in range(len(data)):
        avg.append(np.divide(np.sum(np.array(y_data[i])),3.0))
    return avg

avg_y = average_axis(y_data)

fig, ax1 = plt.subplots()

ax1.set_xlabel('Iterations')
ax1.set_ylabel('Accuracy')

leg = ax1.legend(loc = 'lower right')
ax1.plot(x,y,'b')

ax2 = ax1.twinx()
ax2.plot(x, y1,'r')
ax2.plot(x,y2,'g')

ax2.set_ylabel('Fitness')

red_patch = mpatches.Patch(color='red', label='Mean Test Fitness')
blue_patch = mpatches.Patch(color='blue', label='Root Mean Squared Error')
green_patch = mpatches.Patch(color='green', label='Mean Train Fitness')

plt.legend(handles=[red_patch, blue_patch, green_patch])
plt.show()

plt.show()
