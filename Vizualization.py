import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

import imutils
import os
import sys
import random
import pickle

from imageCorrection import *


def playViz(var, frameset):
    l = openVar(var)
    coordinates = np.array(l)

    # fig2 = plt.figure()
    coordx = coordinates[:, 0]
    coordy = coordinates[:, 1]
    maxX = np.max(coordx) + 10
    maxY = np.max(coordy) + 10

    plt.ion()
    fig, ax = plt.subplots()
    x, y = [], []
    sc = ax.scatter(x, y)
    plt.xlim(-maxX, maxX)
    plt.ylim(-maxY, maxY)

    plt.draw()
    for i in range(len(coordx)):

        x.append(coordx[i])
        y.append(coordy[i])
        sc.set_offsets(np.c_[x, y])
        fig.canvas.draw_idle()
        plt.pause(0.0001)
        cv2.imshow('frame', frameset[i])
        #
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


def showTracking(var):
    l = openVar(var)
    l = np.array(l)
    plt.scatter(l[:, 0], l[:, 1], color='r')
    plt.scatter(l[0, 0], l[0, 1], color='g')
    plt.scatter(l[len(l) - 1, 0], l[len(l) - 1, 1], color='b')

    plt.waitforbuttonpress()
    print('Please click on the graph to close the function')
