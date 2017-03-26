import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

file_name_1 = "mnist-1500-50.txt"
file_name_2 = "cifar-1500-50.txt"

def get_vals(file_name):
    x_data = []
    y_data = []
    with open(file_name, "r") as f:
        for line in f:
            line_split = line.split(":")
            y_data.append([float(line_split[1]),float(line_split[2]),float(line_split[3].strip())])


    with open(file_name, "r") as f:
        for line in f:
            x_data.append(int(line.split(":")[0]))

    def average_axis(data):
        avg = []
        for i in range(len(data)):
            avg.append(np.divide(np.sum(np.array(y_data[i])),3.0))
        return avg

    return x_data, average_axis(y_data)

x,y1 = get_vals(file_name_1)
_,y2 = get_vals(file_name_2)

fig, ax1 = plt.subplots()

ax1.set_title('MNIST and CIFAR, 1500 Iterations')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Accuracy')

leg = ax1.legend(loc = 'bottom right')
ax1.plot(x,y1,'r')
ax1.plot(x,y2,'b')

red_patch = mpatches.Patch(color='red', label='MNIST')
blue_patch = mpatches.Patch(color='blue', label='CIFAR')

plt.legend(loc='lower right',handles=[red_patch, blue_patch])
plt.show()
