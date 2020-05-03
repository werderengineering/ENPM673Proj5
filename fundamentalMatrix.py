from __main__ import *
import numpy as np
import cv2
from test2 import *
from sklearn.preprocessing import normalize


def getrand8(p1, p2):
    indexLin = np.arange(0, len(p1), 1)
    np.random.shuffle(indexLin)

    Indx8 = indexLin[0:8]

    p1 = p1[Indx8, :]
    p2 = p2[Indx8, :]

    return p1, p2


def enforceRank(var):
    u, s, v = np.linalg.svd(var)
    s[-1] = 0
    F = np.dot(u, np.dot(np.diag(s), v)).T
    # if F[2,2]!=1:
    #     print('WARNING BAD F: \n',F)

    return F / F[2, 2]


def createFund(p1, p2):
    n = p1.shape[0]

    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [p1[i][0] * p2[i][0],
                p1[i][0] * p2[i][1],
                p1[i][0],
                p1[i][1] * p2[i][0],
                p1[i][1] * p2[i][1],
                p1[i][1],
                p2[i][0],
                p2[i][1], 1]

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    # if np.any(np.abs(p1-p2))>5:
    #     print (p1,p2)
    #
    # A = np.zeros((len(p1), 9))
    # for i in range(0, len(p1)):
    #

    # x_1 = p1[i][0]
    # y_1 = p1[i][1]
    # x_2 = p2[i][0]
    # y_2 = p2[i][1]
    #
    # A[i] = np.array([x_1 * x_2, x_2 * y_1, x_2, y_2 * x_1, y_2 * y_1, y_2, x_1, y_1, 1])

    # A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
    #         x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
    #         x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]
    # u, s, v = np.linalg.svd(A, full_matrices=True)

    # F = v[-1].reshape(3, 3)
    # u1, s1, v1 = np.linalg.svd(F)
    # s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]])
    # F = np.matmul(np.matmul(u1, s2), v1)

    # F = enforceRank(F)

    U, S, V = np.linalg.svd(F)
    S[2] = 0
    S = np.diag(S)
    F = np.dot(U, np.dot(S, V))

    return F


def computeFMatrix(p1, p2):
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

        FC = EstimateFundamentalMatrix(p18, p28)

        Sin = 0
        mask = []
        for j in range(len(p1)):
            X1 = p1H[j]
            X2 = p2H[j]
            # Error = np.matmul(np.matmul(X2.T, FC), X1)
            Error = np.matmul(X2.T, np.matmul(FC, X1))
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

            p1o = p1[maskIn]
            p2o = p2[maskIn]

    p18o, p28o = getrand8(p1o, p2o)

    F = EstimateFundamentalMatrix(p18o, p28o)
    # F = enforceRank(F)
    # print('#######################################################')
    # print('Inliers: ', SinTotal)
    # print('p1 shape: ', p1.shape[0])
    # print('% Inliers: ', 100 * SinTotal / p1.shape[0])
    #
    # print('\nCustom F\n', F)
    # FCV, CV2mask = cv2.findFundamentalMat(p1, p2, cv2.RANSAC)
    #
    # print('\nCV2 F\n', FCV)
    # FMat, _, _ = ransac_fundamental_matrix(p1, p2)
    # print('\nMat F\n', FMat)
    #
    # # print('\nF Diff: \n', np.abs(FCV - F))
    #
    # print('\nCv2 Mask:\n', CV2mask[:100].T)
    # print('\nCust Mask:\n', mask[:100])

    hold = 1

    return F, p1o, p2o
