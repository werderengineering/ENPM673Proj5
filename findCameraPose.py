"""
Created on Thu April 25 13:00:08 2020
@author: Cheng Chen
"""
import numpy as np


flag_DisambiguateCameraPose = True


def homoTransformation(rotation, translation):
    rigidTransformation = np.concatenate((rotation, translation), axis=1)
    rigidTransformation = np.concatenate((rigidTransformation, np.array([[0, 0, 0, 1]])), axis=0)
    return rigidTransformation


def essentialMatrixFromFundamentalMatrix(fundamentalMatrix, cameraIntrinsicMatrix):
    """
    This function compute e essential matrix from F and K.
    As in the case of F matrix computation, the singular values of E are not necessarily
    (1,1,0) due to the noise in K. This can be corrected by reconstructing it with (1,1,0)
    singular values.
    It is important to note that the F is defined in the original image space (i.e. pixel coordinates) whereas E is in
    the normalized image coordinates. Normalized image coordinates have the origin at the optical center of the image.
    Also, relative camera poses between two views can be computed using E matrix. Moreover, F has 7 degrees of freedom
    while E has 5 as it takes camera parameters in account.

    :param fundamentalMatrix: matrix is only an algebraic representation of epipolar geometry and can both geometrically (contructing the epipolar line) and arithematically
    :param cameraIntrinsicMatrix: camera calibration matrix or camera intrinsic matrix
    :return: essentialMatrix: Essential matrix is another 3×3 matrix, but with some additional properties that relates
    the corresponding points assuming that the cameras obeys the pinhole model (unlike fundamental matrix)
    """
    assert type(fundamentalMatrix) == np.ndarray and type(cameraIntrinsicMatrix) == np.ndarray
    assert fundamentalMatrix.shape == (3, 3) and cameraIntrinsicMatrix.shape == (3, 3)
    assert np.linalg.matrix_rank(fundamentalMatrix) == 2, "The rank of Fundamental Matrix should be 2, but got " + str(np.linalg.matrix_rank(fundamentalMatrix)) + "instead"
    Sigma_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) # use to reconstructing esstential matrix singular values to be (1,1,0)
    """compute the crude esstntial matrix"""
    essentialMatrix = np.dot(np.dot(cameraIntrinsicMatrix.T, fundamentalMatrix), cameraIntrinsicMatrix)
    """correct the essential matrix using signal value decompose"""
    U, Sigma, V_transposed = np.linalg.svd(essentialMatrix)
    Sigma = np.diag(Sigma)
    if np.linalg.norm(essentialMatrix - np.matmul(np.matmul(U, Sigma), V_transposed)) > 1:   # check svd works as it supposed to be
        print("Essential Matrix original is \n" + str(essentialMatrix))
        print("U is\n" + str(U) + "\nSigma is\n" + str(Sigma) + "\nV_transposed is\n" + str(V_transposed))
        print("SVD of Essential Matrix product \n" + str(np.matmul(np.matmul(U, Sigma), V_transposed)))
        # raise ValueError("Essential Matrix and SVD of Essential Matrix product doesn't match")
        print("Essential Matrix and SVD of Essential Matrix product doesn't match")
    essentialMatrix = np.matmul(np.matmul(U, Sigma_new), V_transposed)
    return essentialMatrix


def extractCameraPoseFromEssntialMatrix(essentialMatrix):
    """

    :param essentialMatrix: Essential matrix is another 3×3 matrix, but with some additional properties that relates
    the corresponding points assuming that the cameras obeys the pinhole model (unlike fundamental matrix)
    :return:
    """
    assert type(essentialMatrix) == np.ndarray
    assert essentialMatrix.shape == (3, 3)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  #
    U, Sigma, V_transposed = np.linalg.svd(essentialMatrix)
    """There are four camera pose configurations, (C1, R1), (C2, R2), (C3, R3), (C4, R4)"""
    C1 = U[:, 2].reshape(3, 1)
    R1 = np.dot(np.dot(U, W), V_transposed)
    C2 = - C1
    R2 = R1
    C3 = C1
    R3 = np.dot(np.dot(U, W.T), V_transposed)
    C4 = C2
    R4 = R3
    """choose one camera pose and rotation matrix from four"""
    rotationMatrixes = [R1, R2, R3, R4]
    camerPosistions = [C1, C2, C3, C4]

    for i in range(4):
        if (np.linalg.det(rotationMatrixes[i]) < 0):
            rotationMatrixes[i] = -rotationMatrixes[i]
            camerPosistions[i] = -camerPosistions[i]
    assert R1.shape == (3, 3) and C1.shape == (3, 1)
    return rotationMatrixes, camerPosistions


def linearTriangulation(point_1, projectionMatrix_1, point_2, projectionMatrix_2):
    """
    Triangulate the 3D points, given two camera poses and correpsounding projection matrix.
    :param point_1: np,ndarray <int> first point of pair in camera coordinates
    :param projectionMatrix_1: np,ndarray <float> projection matrix that map the first 2D point to 3D
    :param point_2: np,ndarray <int> second point of pair in camera coordinates
    :param projectionMatrix_2: np,ndarray <float> projection matrix that map the first 2D point to 3D
    :return: the 3D point that this pair corresponding to
    """
    assert type(point_1) == type(point_2) == type(projectionMatrix_1) == type(projectionMatrix_2) == np.ndarray
    assert point_1.shape == point_2.shape == (2, ) or point_1.shape == point_2.shape == (2, 1), "point_1 shape: " + str(point_1.shape) + " point_2 shape: " + str(point_2.shape)
    assert projectionMatrix_1.shape == projectionMatrix_2.shape == (3, 4)
    point_1 = point_1.reshape((2, 1))
    point_2 = point_2.reshape((2, 1))
    D_matrix_1 = (point_1[0,:]* projectionMatrix_1[2,:] - projectionMatrix_1[0,:]).reshape(1, 4)
    D_matrix_2 = (point_1[1,:]* projectionMatrix_1[2,:] - projectionMatrix_1[1,:]).reshape(1, 4)
    D_matrix_3 = (point_2[0,:]* projectionMatrix_2[2,:] - projectionMatrix_2[0,:]).reshape(1, 4)
    D_matrix_4 = (point_2[1,:]* projectionMatrix_2[2,:] - projectionMatrix_2[1,:]).reshape(1, 4)
    D_matrix = np.concatenate((D_matrix_1, D_matrix_2, D_matrix_3, D_matrix_4), axis=0)
    assert D_matrix.shape == (4, 4), "D matrix shape: " + str(D_matrix.shape)
    """slove for the 3D point coordinate"""
    U, Sigma, V_transposed = np.linalg.svd(D_matrix)
    point_3D = V_transposed[-1, :]  # last row of V transposed correspounding to the smallest singular value
    point_3D = point_3D / point_3D[3]
    point_3D = point_3D.reshape(4, 1)
    assert point_3D[3, 0] == 1
    return point_3D


def DisambiguateCameraPose(rigidTransfomation, cameraIntrinsicMatrix, points_firstFrame, points_secondFrame):
    """
    Given a camera pose configurations and their the 2D points correspondences, find how many points pass the cheirality condition
    :param rigidTransfomation:
    :param cameraIntrinsicMatrix:
    :param points_firstFrame:
    :param points_secondFrame:
    :return: number of correspondences pass the cheirality condition
    """
    assert type(points_firstFrame) == type(points_secondFrame) == type(cameraIntrinsicMatrix) == type(rigidTransfomation) == np.ndarray
    assert points_firstFrame.shape[0] == points_secondFrame.shape[0] == 2
    assert cameraIntrinsicMatrix.shape == (3, 3)
    assert rigidTransfomation.shape == (3, 4)
    vote = 0
    rigidTransformation_toSelf = np.concatenate((np.eye(3), np.array([[0],[0],[0]])),axis=1)
    projectionMatrix_firstFrame = np.dot(cameraIntrinsicMatrix, rigidTransformation_toSelf)
    for point_1, point_2 in zip(points_firstFrame.T, points_secondFrame.T):
        projectionMatrix = np.matmul(cameraIntrinsicMatrix, rigidTransfomation)   # form a project matrix of second frame
        point_3D = linearTriangulation(point_1, projectionMatrix_firstFrame, point_2, projectionMatrix)     # shape (4, 1)
        if np.dot(rigidTransfomation[2, 0:3], point_3D[0:3, 0] - rigidTransfomation[2, 3]) > 0:     # the depth (z compound of 3D point)
            vote = vote+1
    print("# of points passed cheirality condition check: " + str(vote))
    return vote


def findSecondFrameCameraPoseFromFirstFrame(fundamentalMatrix, cameraIntrinsicMatrix, points_firstFrame, points_secondFrame):
    """
    Based on the refined fundamental matrix, camera intrinsic matrix, and some points correspondences, find the camera
    pose of first frame from the second frame
    :param fundamentalMatrix: matrix is only an algebraic representation of epipolar geometry and can both geometrically (contructing the epipolar line) and arithematically
    :param cameraIntrinsicMatrix: camera calibration matrix or camera intrinsic matrix
    :param points_firstFrame: a stack of coordinates from first frame, which is the 3D points map to 2D points, and
                            sequentially corresponding to coordinates in second frame
    :param points_secondFrame: a stack of coordinates from second frame, which is the 3D points map to 2D points, and
                            sequentially corresponding to coordinates in first frame
    :return: the rotation and translation matrix between camera where it takes first and where it takes second frame
    """
    assert type(fundamentalMatrix) == type(cameraIntrinsicMatrix) == type(points_firstFrame) == type(points_secondFrame) == np.ndarray
    assert points_firstFrame.shape[1] == points_firstFrame.shape[1] == 2
    assert fundamentalMatrix.shape == cameraIntrinsicMatrix.shape == (3, 3)
    """format data"""
    points_firstFrame = np.insert(points_firstFrame.T, 3, 1, axis=1)
    points_secondFrame = np.insert(points_secondFrame.T, 3, 1, axis=1)

    """get essential matrix"""
    essentialMatrix = essentialMatrixFromFundamentalMatrix(fundamentalMatrix, cameraIntrinsicMatrix)
    """get four possible rotation matrices and translation matrices"""
    rotationMatrices, camerPosistions = extractCameraPoseFromEssntialMatrix(essentialMatrix)
    """pick one of the four possible rotation matrices and translation matrices, using cheirality condition check"""
    votes = np.zeros(4).astype(int)
    for i, (rotationMatrix, translationMatrix) in enumerate(zip(rotationMatrices, camerPosistions)):
        rigidTransfomation = np.concatenate((rotationMatrix, translationMatrix), axis=1)  # rigid transformation
        votes[i] = DisambiguateCameraPose(rigidTransfomation, cameraIntrinsicMatrix, points_firstFrame, points_secondFrame)
    correct_index = votes.argmax()
    return rotationMatrices[correct_index], camerPosistions[correct_index]