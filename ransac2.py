from sklearn import linear_model, datasets
import numpy as np
import matplotlib.pyplot as plt


def ransac2Go(X, Y, p1, p2, p):
    lr = linear_model.LinearRegression()
    lr.fit(X, Y)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X, Y)
    inlier_mask = ransac.inlier_mask_
    # outlier_mask = np.logical_not(inlier_mask)
    #
    # line_X = np.arange(X.min(), X.max())[:, np.newaxis]
    # line_y = lr.predict(line_X)
    # line_y_ransac = ransac.predict(line_X)
    #
    #
    #
    # plt.scatter(X[inlier_mask], Y[inlier_mask], color='yellowgreen', marker='.',
    #             label='Inliers')
    # plt.scatter(X[outlier_mask], Y[outlier_mask], color='gold', marker='.',
    #             label='Outliers')
    #
    # plt.plot(line_X, line_y, color='navy', linewidth=2, label='Linear regressor')
    # plt.plot(line_X, line_y_ransac, color='red', linewidth=2,
    #          label='RANSAC regressor')
    #
    # plt.xlabel("Input")
    # plt.ylabel("Response")
    # plt.show()
    maskTop = inlier_mask[0:len(p1)]
    maskBot = inlier_mask[len(p1):]

    p1 = p1[maskTop.ravel() == 1]
    p2 = p2[maskTop.ravel() == 1]

    # p1 = np.array([p1x[:,0], p1y[:,0]]).T
    # p2 = np.array([p2x[:,0], p2y[:,0]]).T

    return p1, p2


def getrand8(p1, p2):
    indexLin = np.arange(0, len(p1), 1)
    np.random.shuffle(indexLin)

    Indx8 = indexLin[0:100]

    p1 = p1[Indx8, :]
    p2 = p2[Indx8, :]

    return p1, p2
