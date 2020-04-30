from __main__ import *
from imageCorrection import *
import matplotlib.pyplot as plt

l = openVar('visited points')
for i in range(len(l)):
    plt.scatter(l[i][0], l[i][1], color='r')

plt.show()
