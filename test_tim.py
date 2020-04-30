import cv2
import numpy as np
import matplotlib.pyplot as plt
from Oxford_dataset.ReadCameraModel import *
from Oxford_dataset.UndistortImage import *
from imageCorrection import *
import random

fx, fy, cx, cy, Gcameraimage, LUT = ReadCameraModel('./Oxford_dataset/model')
# I'm pretty sure this is correct
K = [[fx, 0, cx],
     [0, fy, cy],
     [0, 0, 1]]
K = np.asarray(K)
sift = cv2.xfeatures2d.SIFT_create()


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


def pipeline(img1, img2):
    # find points in both pictures
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # only take good matches
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # F2, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
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
    E = np.matmul(np.matmul(K.T, F), K)
    return E, pts1, pts2


def compute_P_from_E(E, pts1, pts2):
    Cset = []
    Rset = []
    # make sure E is rank 2
    U, S, V = np.linalg.svd(E)
    # not sure about this
    W = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    tp = U[:, 2]
    tn = -tp
    Cset.append(tp)
    Rset.append(np.matmul(U, np.matmul(W, V.T)))
    Cset.append(tn)
    Rset.append(np.matmul(U, np.matmul(W, V.T)))
    Cset.append(tp)
    Rset.append(np.matmul(U, np.matmul(W.T, V.T)))
    Cset.append(tn)
    Rset.append(np.matmul(U, np.matmul(W.T, V.T)))
    for i in range(4):
        if (np.linalg.det(Rset[i]) < 0):
            Cset[i] = -Cset[i]
            Rset[i] = -Rset[i]
    points, R, t, mask = cv2.recoverPose(E, pts1, pts2)
    return Cset, Rset


def LinearTriangulation(K, C1, R1, C2, R2, x1, x2):
    L = x1.shape[0]
    X1 = cv2.convertPointsToHomogeneous(x1)
    X2 = cv2.convertPointsToHomogeneous(x2)
    depth = np.zeros((L, 3))
    for i in range(L):
        C2 = C2.reshape((3, 1))
        P1 = np.concatenate((R1, C1), axis=1)
        P2 = np.concatenate((R2, C2), axis=1)
        M = np.zeros((6, 6))
        M[:3, :4] = P1
        M[3:, :4] = P2
        M[:3, 4] = -X1[i]
        M[3:, 5] = -X2[i]
        U, S, V = np.linalg.svd(M)
        X = V[-1, :4]
        depth[i] = (X / X[3])[0:3]
    return depth


def DisambiguateCameraPose(Cset, Rset, Xset):
    best = 0
    C = Cset[0]
    R = Rset[0]
    X = Xset[0]
    for i in range(4):
        N = Xset.shape[1]
        n = 0
        for j in range(N):
            R1 = Rset[i][2, :]
            X1 = Xset[i][j, :].T
            C1 = Cset[i]
            if np.matmul(R1, (np.subtract(X1, C1))) > 0:
                n = n + 1
        if n > best:
            C = Cset[i]
            R = Rset[i]
            X = Xset[i]
            best = n
    return C, R, X


def logMovement(Center, PltX, PltY, PltZ):
    CX = Center[0]
    CY = Center[1]
    CZ = Center[2]
    PltX.append(CX)
    PltY.append(CY)
    PltZ.append(CZ)
    return PltX, PltY, PltZ


# cap = cv2.VideoCapture('video2.mp4')
img1 = None
P1 = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
plotList = [(0, 0)]
lastPos = [0, 0]
count = 1
# Plot values
PlotX = []
PlotY = []
PlotZ = []
fig = plt.figure()
Tprev = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
Origin = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

print('\nUnpickling video data')
frameset = openVar('CorrectedFrames')
print('\nVideo Unpickled')

for i, frame in enumerate(frameset):
    if i > 19 and i < 120:
        img2 = frame
        cv2.imshow("video", img2)
        cv2.waitKey(5)
        if img1 is not None:
            print("frame:", count)
            E, pts1, pts2 = pipeline(img1, img2)
            Cset, Rset = compute_P_from_E(E, pts1, pts2)
            Xset = []
            for i in range(4):
                z = np.zeros((3, 1))
                eye = np.eye(3)
                Xset.append(LinearTriangulation(K, z, eye, Cset[i], Rset[i], pts1, pts2))
            Xset = np.asarray(Xset)
            Cset = np.asarray(Cset)
            Rset = np.asarray(Rset)
            C, R, X0 = DisambiguateCameraPose(Cset, Rset, Xset)
            # Build T Matrix
            zer = np.array([[0, 0, 0, 1]])
            C = C.reshape((3, 1))
            T = np.hstack((R, C))
            T = np.vstack((T, zer))
            # Sum of all transformations before this one
            Ttotal = np.matmul(Tprev, T)
            # current spot in Origin frame
            SpotMatrix = np.matmul(Ttotal, Origin)
            # XYZ location
            SpotT = SpotMatrix[:, 3]
            # Log the location
            PlotX, PlotY, PlotZ = logMovement(SpotT, PlotX, PlotY, PlotZ)
            # save T
            Tprev = Ttotal
        img1 = img2
        count = count + 1
ax = plt.axes(projection='3d')
ax.scatter3D(PlotX, PlotY, PlotZ, c=PlotZ)
plt.show()
cv2.destroyAllWindows()
cap.release()
