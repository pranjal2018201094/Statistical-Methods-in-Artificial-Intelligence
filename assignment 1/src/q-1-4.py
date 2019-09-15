import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

data = pd.read_csv('train.csv')
zero=data[data["left"]==0]
one=data[data["left"]==1]
atlist=list(data)
atlist.remove("left")

i=atlist[0]
j=atlist[1]
x_axis0 = zero[i]
y_axis0 = zero[j]
x_axis1 = one[i]
y_axis1 = one[j]
plt.scatter(x_axis0, y_axis0, c="red")
plt.scatter(x_axis1, y_axis1, c="blue")
plt.xlabel(i)
plt.ylabel(j)
plt.title('2-D Plot For Training Data ')
plt.show()