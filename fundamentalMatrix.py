from __main__ import *
import numpy as np
import cv2
from ransac2 import *
from sklearn.preprocessing import normalize


def enforceRank(var):
    u, s, v = np.linalg.svd(var)
    s[2] = 0
    F = np.dot(u, np.dot(np.diag(s), v))

    return F / F[2, 2]


def createFund(p1, p2):
    # print(p1-p2)

    A = np.zeros((8, 9))
    for i in range(0, len(p1)):
        x_1 = p1[i][0]
        y_1 = p1[i][1]
        x_2 = p2[i][0]
        y_2 = p2[i][1]
        A[i] = np.array([x_1 * x_2, x_2 * y_1, x_2, y_2 * x_1, y_2 * y_1, y_2, x_1, y_1, 1])
    u, s, v = np.linalg.svd(A, full_matrices=True)
    F = v[-1].reshape(3, 3)
    # u1, s1, v1 = np.linalg.svd(F)
    # s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]])
    # F = np.matmul(np.matmul(u1, s2), v1)

    F = enforceRank(F)

    return F


def computeFMatrix(p1, p2):
    # p1 = p1 / np.linalg.norm(p1)
    # p2 = p2 / np.linalg.norm(p2)

    # p = np.vstack((p1, p2))
    # xvals = np.array([p[:, 0]]).T
    # yvals = np.array([p[:, 1]]).T
    #
    # p1xR = np.array([p1[:, 0]]).T
    # p2xR = np.array([p2[:, 0]]).T

    # p1, p2 = ransac2Go(xvals, yvals, p1, p2, p)

    eta = .5
    SinTotal = 0
    #
    for i in range(200):

        # if i % 100 == 0:
        #     print(i)

        p18, p28 = getrand8(p1, p2)

        H1 = np.ones([len(p1), 1])
        p1H = np.append(p1, H1, axis=1)
        p2H = np.append(p2, H1, axis=1)

        FC = createFund(p18, p28)

        Sin = 0
        mask = []
        for j in range(len(p1)):
            X1 = p1H[j]
            X2 = p2H[j]
            Error = np.matmul(X2, np.matmul(FC, X1.T))
            cV = np.abs(Error)
            if cV < eta:
                Sin += 1
                mask.append(True)
            else:
                mask.append(False)

        if Sin > SinTotal:
            SinTotal = Sin
            maskIn = mask

            F = FC

            # FIX THIS SO YOU GET ALL INLIERS
            p1o = p1[maskIn]
            p2o = p2[maskIn]

    # F = enforceRank(F)
    # print('#######################################################')
    # print('Inliers: ', SinTotal)
    # print('p1 shape: ', p1.shape[0])
    # print('\nCustom F\n', F)
    # FCV, mask = cv2.findFundamentalMat(p1, p2, cv2.RANSAC, eta)
    # print('\nCV2 F\n', FCV)
    #
    # print('\nF Diff: \n', np.abs(FCV - F))

    hold = 1

    return F, p1o, p2o

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
