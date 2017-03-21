import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches

tempArr = []
with open("src/cifar-1500-50.txt") as f:
    for line in f:
        temp = line.strip().split(":")
        tempArr.append(temp)

print(tempArr)

y = []
y1 = []
y2 = []
x= []


for i in range(0, len(tempArr)):
    total = total1 = total2 = 0
    total += math.pow(float(tempArr[i][2]),2) + math.pow(float(tempArr[i + 1][2]),2) + math.pow(float(tempArr[i + 2][2]),2)+ math.pow(float(tempArr[i + 3][2]),2) + math.pow(float(tempArr[i + 4][2]),2)
    total1 += float(tempArr[i][1]) + float(tempArr[i + 1][1]) + float(tempArr[i + 2][1])+ float(tempArr[i + 3][1]) + float(tempArr[i + 4][1])
    total2 += float(tempArr[i][0]) + float(tempArr[i + 1][0]) + float(tempArr[i + 2][0])+ float(tempArr[i + 3][0]) + float(tempArr[i + 4][0])

    root_mean_sq = math.sqrt(total / 5)
    mean1 = total1 / 5
    mean2 = total2 / 5
    y1.append(mean1)
    y2.append(mean2)
    x.append(tempArr[i][3])
    y.append(root_mean_sq)

fig, ax1 = plt.subplots()

ax1.set_xlabel('Training Size (%)')
ax1.set_ylabel('RMSE')

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
