import cv2
import numpy as np
import matplotlib.pyplot as plt
from Oxford_dataset.ReadCameraModel import *
from Oxford_dataset.UndistortImage import *
from imageCorrection import *
from numpy import vstack, diag, std, dot, mean, array, linalg, linspace, sqrt
import random

import numpy as np
import sys

sys.dont_write_bytecode = True


#
# fx, fy, cx, cy, Gcameraimage, LUT = ReadCameraModel('./Oxford_dataset/model')
# # I'm pretty sure this is correct
# K = [[fx, 0, cx],
#      [0, fy, cy],
#      [0, 0, 1]]
# K = np.asarray(K)
# sift = cv2.xfeatures2d.SIFT_create()
#
#
def compute_fundamental(x1, x2):
    # this is for comparison against our F
    n = x1.shape[0]
    # build matrix for equations
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[i][0] * x2[i][0],
                x1[i][0] * x2[i][1],
                x1[i][0],
                x1[i][1] * x2[i][0],
                x1[i][1] * x2[i][1],
                x1[i][1],
                x2[i][0],
                x2[i][1], 1]
    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    S = np.diag(S)
    F = np.dot(U, np.dot(S, V))
    return F


#
#
# def pipeline(img1, img2):
#     # find points in both pictures
#     kp1, des1 = sift.detectAndCompute(img1, None)
#     kp2, des2 = sift.detectAndCompute(img2, None)
#     FLANN_INDEX_KDTREE = 0
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)
#     good = []
#     pts1 = []
#     pts2 = []
#     # only take good matches
#     for i, (m, n) in enumerate(matches):
#         if m.distance < 0.8 * n.distance:
#             good.append(m)
#             pts2.append(kp2[m.trainIdx].pt)
#             pts1.append(kp1[m.queryIdx].pt)
#     pts1 = np.int32(pts1)
#     pts2 = np.int32(pts2)
#     # F2, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
#     n = 0
#     itters = 50
#     eps = .6
#     sin = []
#     L = len(pts1) - 1
#     x1 = np.asarray(pts1)
#     x2 = np.asarray(pts2)
#     for i in range(0, itters):
#         indicies = [random.randint(0, L), random.randint(0, L), random.randint(0, L), random.randint(0, L),
#                     random.randint(0, L)
#             , random.randint(0, L), random.randint(0, L), random.randint(0, L)]
#         x1hat = x1[indicies]
#         x2hat = x2[indicies]
#         F = compute_fundamental(x1hat, x2hat)
#         s = []
#         for j in range(0, L):
#             x1j = np.asarray([x1[j][0], x1[j][1], 1])
#             x2j = np.asarray([x2[j][0], x2[j][1], 1])
#             mult = np.matmul(x2j.T, np.matmul(F, x1j))
#             if abs(mult) < eps:
#                 s.append(j)
#         if len(s) > n:
#             n = len(s)
#             sin = s
#     x1new = x1[sin]
#     x2new = x2[sin]
#     F = compute_fundamental(x1new, x2new)
#     E = np.matmul(np.matmul(K.T, F), K)
#     return E, pts1, pts2
#
#
# def compute_P_from_E(E, pts1, pts2):
#     Cset = []
#     Rset = []
#     # make sure E is rank 2
#     U, S, V = np.linalg.svd(E)
#     # not sure about this
#     W = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
#     tp = U[:, 2]
#     tn = -tp
#     Cset.append(tp)
#     Rset.append(np.matmul(U, np.matmul(W, V.T)))
#     Cset.append(tn)
#     Rset.append(np.matmul(U, np.matmul(W, V.T)))
#     Cset.append(tp)
#     Rset.append(np.matmul(U, np.matmul(W.T, V.T)))
#     Cset.append(tn)
#     Rset.append(np.matmul(U, np.matmul(W.T, V.T)))
#     for i in range(4):
#         if (np.linalg.det(Rset[i]) < 0):
#             Cset[i] = -Cset[i]
#             Rset[i] = -Rset[i]
#     points, R, t, mask = cv2.recoverPose(E, pts1, pts2)
#     return Cset, Rset
#
#
# def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
#     L = x1.shape[0]
#     X1 = cv2.convertPointsToHomogeneous(x1)
#     X2 = cv2.convertPointsToHomogeneous(x2)
#     depth = np.zeros((L, 3))
#     for i in range(L):
#         C2 = C2.reshape((3, 1))
#         P1 = np.concatenate((R1, C1), axis=1)
#         P2 = np.concatenate((R2, C2), axis=1)
#         M = np.zeros((6, 6))
#         M[:3, :4] = P1
#         M[3:, :4] = P2
#         M[:3, 4] = -X1[i]
#         M[3:, 5] = -X2[i]
#         U, S, V = np.linalg.svd(M)
#         X = V[-1, :4]
#         depth[i] = (X / X[3])[0:3]
#     return depth
#
#
# def DisambiguateCameraPose(Cset, Rset, Xset):
#     best = 0
#     C = Cset[0]
#     R = Rset[0]
#     X = Xset[0]
#     for i in range(4):
#         N = Xset.shape[1]
#         n = 0
#         for j in range(N):
#             R1 = Rset[i][2, :]
#             X1 = Xset[i][j, :].T
#             C1 = Cset[i]
#             if np.matmul(R1, (np.subtract(X1, C1))) > 0:
#                 n = n + 1
#         if n > best:
#             C = Cset[i]
#             R = Rset[i]
#             X = Xset[i]
#             best = n
#     return C, R, X
#
#
# def logMovement(Center, PltX, PltY, PltZ):
#     CX = Center[0]
#     CY = Center[1]
#     CZ = Center[2]
#     PltX.append(CX)
#     PltY.append(CY)
#     PltZ.append(CZ)
#     return PltX, PltY, PltZ
#
#
# # cap = cv2.VideoCapture('video2.mp4')
# img1 = None
# P1 = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
# plotList = [(0, 0)]
# lastPos = [0, 0]
# count = 1
# # Plot values
# PlotX = []
# PlotY = []
# PlotZ = []
# fig = plt.figure()
# Tprev = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
# Origin = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#
# print('\nUnpickling video data')
# frameset = openVar('CorrectedFrames')
# print('\nVideo Unpickled')
#
# for i, frame in enumerate(frameset):
#     if i > 19 and i < 120:
#         img2 = frame
#         cv2.imshow("video", img2)
#         cv2.waitKey(5)
#         if img1 is not None:
#             print("frame:", count)
#             E, pts1, pts2 = pipeline(img1, img2)
#             Cset, Rset = compute_P_from_E(E, pts1, pts2)
#             Xset = []
#             for i in range(4):
#                 z = np.zeros((3, 1))
#                 eye = np.eye(3)
#                 Xset.append(LinearTriangulation(K, z, eye, Cset[i], Rset[i], pts1, pts2))
#             Xset = np.asarray(Xset)
#             Cset = np.asarray(Cset)
#             Rset = np.asarray(Rset)
#             C, R, X0 = DisambiguateCameraPose(Cset, Rset, Xset)
#             # Build T Matrix
#             zer = np.array([[0, 0, 0, 1]])
#             C = C.reshape((3, 1))
#             T = np.hstack((R, C))
#             T = np.vstack((T, zer))
#             # Sum of all transformations before this one
#             Ttotal = np.matmul(Tprev, T)
#             # current spot in Origin frame
#             SpotMatrix = np.matmul(Ttotal, Origin)
#             # XYZ location
#             SpotT = SpotMatrix[:, 3]
#             # Log the location
#             PlotX, PlotY, PlotZ = logMovement(SpotT, PlotX, PlotY, PlotZ)
#             # save T
#             Tprev = Ttotal
#         img1 = img2
#         count = count + 1
# ax = plt.axes(projection='3d')
# ax.scatter3D(PlotX, PlotY, PlotZ, c=PlotZ)
# plt.show()
# cv2.destroyAllWindows()
# cap.release()


""" File to implement function to calculate Fundamental Matrix
"""


def EstimateFundamentalMatrix(points_a, points_b):
    points_a = np.asarray(points_a)
    points_b = np.asarray(points_b)
    """Function to calculate Fundamental Matrix
    Args:
        points_a (list): List of points from image 1
        points_b (list): List of points from image 2
    Returns:
        array: Fundamental Matrix
    """
    points_num = points_a.shape[0]
    A = []
    B = np.ones((points_num, 1))

    cu_a = np.sum(points_a[:, 0]) / points_num
    cv_a = np.sum(points_a[:, 1]) / points_num

    s = points_num / np.sum(
        ((points_a[:, 0] - cu_a) ** 2 + (points_a[:, 1] - cv_a) ** 2) ** (1 / 2))
    T_a = np.dot(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -cu_a], [0, 1, -cv_a], [0, 0, 1]]))

    points_a = np.array(points_a.T)
    points_a = np.append(points_a, B)

    points_a = np.reshape(points_a, (3, points_num))
    points_a = np.dot(T_a, points_a)
    points_a = points_a.T

    cu_b = np.sum(points_b[:, 0]) / points_num
    cv_b = np.sum(points_b[:, 1]) / points_num

    s = points_num / np.sum(
        ((points_b[:, 0] - cu_b) ** 2 + (points_b[:, 1] - cv_b) ** 2) ** (1 / 2))
    T_b = np.dot(
        np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]]),
        np.array([[1, 0, -cu_b], [0, 1, -cv_b], [0, 0, 1]]))

    points_b = np.array(points_b.T)
    points_b = np.append(points_b, B)

    points_b = np.reshape(points_b, (3, points_num))
    points_b = np.dot(T_b, points_b)
    points_b = points_b.T

    for i in range(points_num):
        u_a = points_a[i, 0]
        v_a = points_a[i, 1]
        u_b = points_b[i, 0]
        v_b = points_b[i, 1]
        A.append([
            u_a * u_b, v_a * u_b, u_b, u_a * v_b, v_a * v_b, v_b, u_a, v_a, 1
        ])

    #     A = np.array(A)
    #     F = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, -B))
    #     F = np.append(F,[1])

    _, _, v = np.linalg.svd(A)
    F = v[-1]

    F = np.reshape(F, (3, 3)).T
    F = np.dot(T_a.T, F)
    F = np.dot(F, T_b)

    F = F.T
    U, S, V = np.linalg.svd(F)
    S = np.array([[S[0], 0, 0], [0, S[1], 0], [0, 0, 0]])
    F = np.dot(U, S)
    F = np.dot(F, V)

    F = F / F[2, 2]

    return F, points_a, points_a


def MatFund(pts1, pts2):
    n = 0
    itters = 50
    eps = .6
    sin = []
    L = len(pts1) - 1
    x1 = np.asarray(pts1)
    x2 = np.asarray(pts2)
    for i in range(0, itters):
        indicies = [random.randint(0, L), random.randint(0, L), random.randint(0, L), random.randint(0, L),
                    random.randint(0, L)
            , random.randint(0, L), random.randint(0, L), random.randint(0, L)]
        x1hat = x1[indicies]
        x2hat = x2[indicies]
        F = compute_fundamental(x1hat, x2hat)
        s = []
        for j in range(0, L):
            x1j = np.asarray([x1[j][0], x1[j][1], 1])
            x2j = np.asarray([x2[j][0], x2[j][1], 1])
            mult = np.matmul(x2j.T, np.matmul(F, x1j))
            if abs(mult) < eps:
                s.append(j)
        if len(s) > n:
            n = len(s)
            sin = s
    x1new = x1[sin]
    x2new = x2[sin]
    F = compute_fundamental(x1new, x2new)

    return F, x1new, x2new


def compute_P(x, X):
    """    Compute camera matrix from pairs of
        2D-3D correspondences (in homog. coordinates). """

    n = x.shape[1]
    if X.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # create matrix for DLT solution
    M = zeros((3 * n, 12 + n))
    for i in range(n):
        M[3 * i, 0:4] = X[:, i]
        M[3 * i + 1, 4:8] = X[:, i]
        M[3 * i + 2, 8:12] = X[:, i]
        M[3 * i:3 * i + 3, i + 12] = -x[:, i]

    U, S, V = linalg.svd(M)

    return V[-1, :12].reshape((3, 4))


def triangulate_point(x1, x2, P1, P2):
    """ Point pair triangulation from
        least squares solution. """

    M = zeros((6, 6))
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -x1
    M[3:, 5] = -x2

    U, S, V = linalg.svd(M)
    X = V[-1, :4]

    return X / X[3]


def triangulate(x1, x2, P1, P2):
    """    Two-view triangulation of points in
        x1,x2 (3*n homog. coordinates). """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]
    return array(X).T


def compute_fundamental(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    # compute linear least square solution
    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = linalg.svd(F)
    S[2] = 0
    F = dot(U, dot(diag(S), V))

    return F / F[2, 2]


def compute_epipole(F):
    """ Computes the (right) epipole from a
        fundamental matrix F.
        (Use with F.T for left epipole.) """

    # return null space of F (Fx=0)
    U, S, V = linalg.svd(F)
    e = V[-1]
    return e / e[2]


def plot_epipolar_line(im, F, x, epipole=None, show_epipole=True):
    """ Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix
        and x a point in the other image."""

    m, n = im.shape[:2]
    line = dot(F, x)

    # epipolar line parameter and values
    t = linspace(0, n, 100)
    lt = array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt >= 0) & (lt < m)
    plot(t[ndx], lt[ndx], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """

    return array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def compute_P_from_fundamental(F):
    """    Computes the second camera matrix (assuming P1 = [I 0])
        from a fundamental matrix. """

    e = compute_epipole(F.T)  # left epipole
    Te = skew(e)
    return vstack((dot(Te, F.T).T, e)).T


def compute_P_from_essential(E):
    """    Computes the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. Output is a list of four
        possible camera matrices. """

    # make sure E is rank 2
    U, S, V = svd(E)
    if det(dot(U, V)) < 0:
        V = -V
    E = dot(U, dot(diag([1, 1, 0]), V))

    # create matrices (Hartley p 258)
    Z = skew([0, 0, -1])
    W = array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # return all four solutions
    P2 = [vstack((dot(U, dot(W, V)).T, U[:, 2])).T,
          vstack((dot(U, dot(W, V)).T, -U[:, 2])).T,
          vstack((dot(U, dot(W.T, V)).T, U[:, 2])).T,
          vstack((dot(U, dot(W.T, V)).T, -U[:, 2])).T]

    return P2


def compute_fundamental_normalized(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the normalized 8 point algorithm. """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = mean(x1[:2], axis=1)
    S1 = sqrt(2) / std(x1[:2])
    T1 = array([[S1, 0, -S1 * mean_1[0]], [0, S1, -S1 * mean_1[1]], [0, 0, 1]])
    x1 = dot(T1, x1)

    x2 = x2 / x2[2]
    mean_2 = mean(x2[:2], axis=1)
    S2 = sqrt(2) / std(x2[:2])
    T2 = array([[S2, 0, -S2 * mean_2[0]], [0, S2, -S2 * mean_2[1]], [0, 0, 1]])
    x2 = dot(T2, x2)

    # compute F with the normalized coordinates
    F = compute_fundamental(x1, x2)

    # reverse normalization
    F = dot(T1.T, dot(F, T2))

    return F / F[2, 2]


class RansacModel(object):
    """ Class for fundmental matrix fit with ransac.py from
        http://www.scipy.org/Cookbook/RANSAC"""

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """ Estimate fundamental matrix using eight
            selected correspondences. """

        # transpose and split data into the two point sets
        data = data.T
        x1 = data[:3, :8]
        x2 = data[3:, :8]

        # estimate fundamental matrix and return
        F = compute_fundamental_normalized(x1, x2)
        return F

    def get_error(self, data, F):
        """ Compute x^T F x for all correspondences,
            return error for each transformed point. """

        # transpose and split data into the two point
        data = data.T
        x1 = data[:3]
        x2 = data[3:]

        # Sampson distance as error measure
        Fx1 = dot(F, x1)
        Fx2 = dot(F, x2)
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        err = (diag(dot(x1.T, dot(F, x2)))) ** 2 / denom

        # return error per point
        return err


def F_from_ransac(x1, x2, model, maxiter=5000, match_theshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).
        input: x1,x2 (3*n arrays) points in hom. coordinates. """

    import ransac

    data = vstack((x1, x2))

    # compute F and return with inlier index
    F, ransac_data = ransac.ransac(data.T, model, 8, maxiter, match_theshold, 20, return_all=True)
    return F, ransac_data['inliers']
