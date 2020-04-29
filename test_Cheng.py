import numpy as np
import findCameraPose

def a():
    a = np.array([[0, 0, 0], [1, 1, 1]]).T
    print(a)
    b = np.array([[1, 1, 1], [2, 2, 2]]).T
    print(b)
    c = np.concatenate((a, b), axis=1)
    print(c)
    c = np.insert(c, 4, 5, axis=1)
    print(c)


def b():
    votes = np.ones(4).astype(int)
    print(votes)


def c():
    points_firstFrame = np.array([[0, 0],[1, 1],[2, 2],[3, 3],[4, 4]])
    points_secondFrame = points_firstFrame + 1
    fundamental = np.eye(3)
    fundamental[0, 2] = 1
    fundamental[1, 2] = 1
    fundamental[2, 2] = 0
    camera = np.eye(3)
    R, t = findCameraPose.findSecondFrameCameraPoseFromFirstFrame(fundamental, camera, points_firstFrame, points_secondFrame)
    print(R)
    print(t)

c()