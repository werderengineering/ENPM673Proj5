from __main__ import *
import numpy as np
import cv2


def randomMatchingPoints(matchpts1, matchpts2):
    rand_index = np.random.randint(len(matchpts1), size=8)

    X1 = np.array(
        [matchpts1[rand_index[0]], matchpts1[rand_index[1]], matchpts1[rand_index[2]], matchpts1[rand_index[3]],
         matchpts1[rand_index[4]], matchpts1[rand_index[5]], matchpts1[rand_index[6]], matchpts1[rand_index[7]]])
    X2 = np.array(
        [matchpts2[rand_index[0]], matchpts2[rand_index[1]], matchpts2[rand_index[2]], matchpts2[rand_index[3]],
         matchpts2[rand_index[4]], matchpts2[rand_index[5]], matchpts2[rand_index[6]], matchpts2[rand_index[7]]])

    return X1, X2


def NormalizedFMatrix(p1, p2):
    dist1 = np.sqrt((p1[:, 0] - np.mean(p1[:, 0])) ** 2 + (p1[:, 1] - np.mean(p1[:, 1])) ** 2)
    dist2 = np.sqrt((p2[:, 0] - np.mean(p2[:, 0])) ** 2 + (p2[:, 1] - np.mean(p2[:, 1])) ** 2)

    m_dist1 = np.mean(dist1)
    m_dist2 = np.mean(dist2)

    scale1 = np.sqrt(2) / m_dist1
    scale2 = np.sqrt(2) / m_dist2

    t1 = np.array(
        [[scale1, 0, -scale1 * np.mean(p1[:, 0])], [0, scale1, -scale1 * np.mean(p1[:, 1])], [0, 0, 1]])
    t2 = np.array(
        [[scale2, 0, -scale2 * np.mean(p2[:, 0])], [0, scale2, -scale2 * np.mean(p2[:, 1])], [0, 0, 1]])

    U_x = (p1[:, 0] - np.mean(p1[:, 0])) * scale1
    U_y = (p1[:, 1] - np.mean(p1[:, 1])) * scale1
    V_x = (p2[:, 0] - np.mean(p2[:, 0])) * scale2
    V_y = (p2[:, 1] - np.mean(p2[:, 1])) * scale2

    A = np.zeros((len(U_x), 9))

    for i in range(len(U_x)):
        A[i] = np.array(
            [U_x[i] * V_x[i], U_y[i] * V_x[i], V_x[i], U_x[i] * V_y[i], U_y[i] * V_y[i], V_y[i], U_x[i], U_y[i], 1])

    U, S, V = np.linalg.svd(A)
    V = V.T
    F = V[:, -1].reshape(3, 3)

    Uf, Sf, Vf = np.linalg.svd(F)
    SF = np.diag(Sf)
    SF[2, 2] = 0

    F = Uf @ SF @ Vf
    F = t2.T @ F @ t1
    F = F / F[2, 2]

    return F


def AdFund(U, V):
    Inliers_UN = []
    max_inliers = 0
    for i in range(200):
        if i % 100 == 0:
            print(i)
        X1, X2 = randomMatchingPoints(U, V)
        F_r = NormalizedFMatrix(X1, X2)
        Inliers_U = []
        Inliers_V = []
        Inliers = 0
        for j in range(len(U)):
            U1 = np.array([U[j][0], U[j][1], 1]).reshape(1, -1)
            V1 = np.array([V[j][0], V[j][1], 1]).reshape(1, -1)

            epiline1 = F_r @ U1.T
            epiline2 = F_r.T @ V1.T
            error_bottom = epiline1[0] ** 2 + epiline1[1] ** 2 + epiline2[0] ** 2 + epiline2[1] ** 2

            error = ((V1 @ F_r @ U1.T) ** 2) / error_bottom

            if error[0, 0] < .008:
                Inliers += 1
                Inliers_U.append([U[j][0], U[j][1]])
                Inliers_V.append([V[j][0], V[j][1]])

        if max_inliers < Inliers:
            max_inliers = Inliers
            Inliers_UN = Inliers_U
            Inliers_VN = Inliers_V
            F = F_r

    Inliers_UN = np.array(Inliers_UN)
    Inliers_VN = np.array(Inliers_VN)

    return F, Inliers_UN, Inliers_VN


def AdMatches(img_current_frame, img_next_frame):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_current_frame, None)
    kp2, des2 = sift.detectAndCompute(img_next_frame, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)

    U = []
    V = []
    for m in matches:
        pts_1 = kp1[m.queryIdx]
        x1, y1 = pts_1.pt
        pts_2 = kp2[m.trainIdx]
        x2, y2 = pts_2.pt
        U.append((x1, y1))
        V.append((x2, y2))
    U = np.array(U)
    V = np.array(V)
    return U, V
