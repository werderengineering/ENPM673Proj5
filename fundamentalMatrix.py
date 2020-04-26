from __main__ import *
import numpy as np
from ransac import *


def enforceRank(var):
    u, s, v = np.linalg.svd(var)
    s[2] = 0
    return np.dot(u, np.dot(np.diag(s), v))


def computeFMatrix(p1, p2):
    #
    # p1 = p1 / np.linalg.norm(p1)
    # p2 = p2 / np.linalg.norm(p2)

    p = np.vstack((p1, p2))

    p1, p2 = ransacGo(p[:,0], p[:,1],p1[:,0],p2[:,0])

    A = np.zeros((p1.shape[1], 9))
    for i in range(p1.shape[1]):
        A[i] = [
            p2[0, i] * p1[0, i],
            p2[0, i] * p1[1, i],
            p2[0, i],
            p2[1, i] * p1[0, i],
            p2[1, i] * p1[1, i],
            p2[1, i],
            p1[0, i],
            p2[1, i],
            1
        ]


    u, s, v = np.linalg.svd(A)
    F = v[-1].reshape(3, 3)

    F=enforceRank(F)


    return F


# p1 = np.array([
#     [0, 1],
#     [1, 1],
#     [2, 5],
#     [3, 5],
#     [4, 1],
#     [5, 1],
#     [5, 5],
#     [6, 5],
# ])
#
# p2 = p1 + 2
#
# F = computeFMatrix(p1, p2)
# print(F)

