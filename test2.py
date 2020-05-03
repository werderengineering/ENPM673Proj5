import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:
                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]
    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.
    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_3d: A numpy array of shape (N, 3)
    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    ###########################################################################
    ###########################################################################
    b = np.zeros(((int(2 * points_3d.shape[0])), 1))
    A = np.zeros((int(2 * points_3d.shape[0]), 12))
    x = 0
    for i in range(0, points_3d.shape[0]):
        A[x, 0:3] = points_3d[i, :]
        A[x, 3] = 1
        A[x, 8:11] = -points_2d[i, 0] * points_3d[i, :]
        A[x, 11] = -points_2d[i, 0]
        A[x + 1, 4:7] = points_3d[i, :]
        A[x + 1, 7] = 1
        A[x + 1, 8:11] = -points_2d[i, 1] * points_3d[i, :]
        A[x + 1, 11] = -points_2d[i, 1]
        x = x + 2
    U, S, VT = np.linalg.svd(A)
    V = VT.T
    Mtemp = V[:, V.shape[1] - 1]
    M = np.zeros((3, 4))
    M[0, :] = Mtemp[:4]
    M[1, :] = Mtemp[4:8]
    M[2, :] = Mtemp[8:12]
    ###########################################################################
    ###########################################################################

    return M


def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.
    The center of the camera C can be found by:
        C = -Q^(-1)m4
    where your project matrix M = (Q | m4).
    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.

    ###########################################################################
    ###########################################################################
    Q = M[0:3, 0:3]
    Qinv = np.linalg.inv(Q)
    cc = np.matmul(-Qinv, M[:, 3])
    ###########################################################################
    ###########################################################################
    return cc


def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.
    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.
    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B
    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    ###########################################################################
    ###########################################################################
    N = points_a.shape[0]

    m = np.average(points_a, axis=0)
    m_mean = points_a - m.reshape(1, 2)
    s_sum = np.sum((m_mean) ** 2, axis=None)
    sinv = 1 / ((s_sum / (2 * N)) ** 0.5)
    x = m_mean * sinv

    mdash = np.average(points_b, axis=0)
    mdash_mean = points_b - mdash.reshape(1, 2)
    sdash_sum = np.sum((mdash_mean) ** 2, axis=None)
    sdinv = 1 / ((sdash_sum / (2 * N)) ** 0.5)
    y = mdash_mean * sdinv

    A = np.ones((N, 9))
    A[:, 0:2] = x * y[:, 0].reshape(N, 1)
    A[:, 2] = y[:, 0]
    A[:, 3:5] = x * y[:, 1].reshape(N, 1)
    A[:, 5] = y[:, 1]
    A[:, 6:8] = x

    u, s, vt = np.linalg.svd(Y, full_matrices=True)
    F = vt[8, :].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F, full_matrices=True)
    S[2] = 0
    Smat = np.diag(S)
    F = np.dot(U, np.dot(Smat, Vt))

    T = np.zeros((3, 3))
    T[0, 0] = sinv
    T[1, 1] = sinv
    T[2, 2] = 1
    T[0, 2] = -sinv * m[0]
    T[1, 2] = -sinv * m[1]

    Tdash = np.zeros((3, 3))
    Tdash[0, 0] = sdinv
    Tdash[1, 1] = sdinv
    Tdash[2, 2] = 1
    Tdash[0, 2] = -sdinv * mdash[0]
    Tdash[1, 2] = -sdinv * mdash[1]

    F = np.dot(np.transpose(Tdash), np.dot(F, T))
    ###########################################################################
    ###########################################################################

    return F


def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.
    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.
    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)
    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    # best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    # inliers_a = matches_a[:100, :]
    # inliers_b = matches_b[:100, :]

    ###########################################################################
    ###########################################################################
    N = 500
    S = matches_b.shape[0]
    r = np.random.randint(S, size=(N, 8))

    m = np.ones((3, S))
    m[0:2, :] = matches_a.T
    mdash = np.ones((3, S))
    mdash[0:2, :] = matches_b.T
    count = np.zeros(N)
    cost = np.zeros(S)
    t = 1e-2
    for i in range(N):
        # if i%1000==0:
        #     print(i)
        cost1 = np.zeros(8)
        F = estimate_fundamental_matrix(matches_a[r[i, :], :], matches_b[r[i, :], :])
        for j in range(S):
            cost[j] = np.dot(np.dot(mdash[:, j].T, F), m[:, j])
        inlie = np.absolute(cost) < t
        count[i] = np.sum(inlie + np.zeros(S), axis=None)

    index = np.argsort(-count)
    best = index[0]
    best_F = estimate_fundamental_matrix(matches_a[r[best, :], :], matches_b[r[best, :], :])
    for j in range(S):
        cost[j] = np.dot(np.dot(mdash[:, j].T, best_F), m[:, j])
    confidence = np.absolute(cost)
    index = np.argsort(confidence)
    matches_b = matches_b[index]
    matches_a = matches_a[index]

    inliers_a = matches_a[:100, :]
    inliers_b = matches_b[:100, :]

    ###########################################################################
    ###########################################################################

    return best_F, inliers_a, inliers_b


def EstimateFundamentalMatrix(points_a, points_b):
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

    # points_a = np.array(points_a.T)
    points_a = np.append(points_a, B, axis=1)

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

    # points_b = np.array(points_b.T)
    points_b = np.append(points_b, B, axis=1)

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

    return F
