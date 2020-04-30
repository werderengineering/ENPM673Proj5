import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import os
import sys
import random
import pickle

from Oxford_dataset.ReadCameraModel import *
from Oxford_dataset.UndistortImage import *
from imageCorrection import *
from fundamentalMatrix import *
from featureMatch import *
from findCameraPose import *
from comparable import *

# from useBuiltinFunction import *

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = True
prgRun = True
customFun_MatchFeat = False
customFun_fundamentalMatrix = False
customFun_cameraPose = False
customFun_Homo = False


def main(prgRun):
    # start file
    print("Run")
    # Hard input stand-ins
    directory = "./Oxford_dataset/stereo/centre/"
    FixImagery = False
    ###########################
    # Soft inputs
    # This is a pain

    # getdataYN = str.lower(input('Create a data set? Enter |yes| or |no|: '))
    # if getdataYN == 'yes':
    #     FixImagery = True
    #
    # directory = str(input(
    #     'What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows". Example: ./Bolt2/img: \n'))

    ###########################
    fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('./Oxford_dataset/model')
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    Origin = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    PPose = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    PoseList = []
    PoseList.append(Origin[:2, 3])
    pose_initial = np.array([[0], [0], [0], [1]])

    rigidTransformation_fromInitial = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    listPose = []
    homo1 = np.identity(4)
    t1 = np.array([[0, 0, 0, 1]])
    t1 = t1.T

    sift = cv2.xfeatures2d.SIFT_create()

    # fig = plt.figure()
    plt.show()

    if FixImagery:
        correctImages(directory, LUT)

    print('\nUnpickling video data')
    frameset = openVar('CorrectedFrames')
    print('\nVideo Unpickled')

    for i, frame in enumerate(frameset):
        ###########################
        # Add Functions here
        img1 = frame.copy()

        ######
        # Change this value to make th test longer or shorter. Good comparison is 500

        if i > 200:
            break
        if i > 19:

            if i % 50 == 0:
                print(i)
            # 1. Point correspondence
            if customFun_MatchFeat:

                p1, p2 = matchUntil(img2, img1, sift)

            else:
                p1, p2 = getcomparablePoints(img2, img1, sift)

            # 2. Est fund Matrix w/ ransac
            #     2a. Center and scale to 8 point
            #     2b. Est Fund mat via ransac
            #     2c. Enforce rank 2 contraint

            if customFun_fundamentalMatrix:
                F = computeFMatrix(p1, p2)
            else:
                F = getcomparableF(p1, p2)

            # 3. Fin e matrix from F with calibration params
            # 4. Decompe E into T and R
            # 5. Find T and R solutions (cherality) use T and R giving largest  positive depth vals
            if customFun_cameraPose:
                rotation, translation = findSecondFrameCameraPoseFromFirstFrame(F, K, p1, p2)
            else:
                rotation, translation = getcomparableRT(F, K, p1, p2)

                # rotation,translation=poseMatrix(img1,img2,K,sift)

            # 6. plot pos of cam center based on rot and tans
            """start: form rigid body transformation matrix using R and T"""
            if customFun_Homo:
                rigidTransformation_fromLast = homoTransformation(rotation, translation)
                rigidTransformation_fromInitial = rigidTransformation_fromInitial @ rigidTransformation_fromLast
                p1 = np.dot(rigidTransformation_fromInitial, pose_initial)
            else:
                homo2 = getcomprableHomo(rotation, translation)
                homo1 = np.dot(homo1, homo2)
                p1 = np.dot(homo1, t1)
            """end"""

            plt.scatter(p1[0][0], -p1[2][0], color='r')
            listPose.append([p1[0][0], -p1[2][0]])

            ###########################
            # cv2.imshow('img1', img1)
            # cv2.imshow('img2', img2)
            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()

        img2 = img1.copy()

    plt.show()

    prgRun = False
    return prgRun


print('Function Initializations complete')

if __name__ == '__main__':
    print('Start')
    prgRun = True
    while prgRun == True:
        prgRun = main(prgRun)

    print('Goodbye!')
    cv2.destroyAllWindows()
