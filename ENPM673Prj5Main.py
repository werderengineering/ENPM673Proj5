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

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = False
prgRun = True


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

    rigidTransformation_fromInitial = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
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
        if i > 19:
            # 1. Point correspondence
            p1, p2 = MatchFeatures(img1, img2)

            # 2. Est fund Matrix w/ ransac
            #     2a. Center and scale to 8 point
            #     2b. Est Fund mat via ransac
            #     2c. Enforce rank 2 contraint
            # print('\n\nNEW DATA\n\n')
            F = computeFMatrix(p1, p2)
            # F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC)

            # 3. Fin e matrix from F with calibration params
            # EM = essentialMatrixFromFundamentalMatrix(F, K)
            # 4. Decompe E into T and R
            # 5. Find T and R solutions (cherality) use T and R giving largest  positive depth vals
            rotation, translation = findSecondFrameCameraPoseFromFirstFrame(F, K, p1, p2)

            # 6. plot pos of cam center based on rot and tans
            # Position = np.matmul(NPose, Origin)
            # PoseList.append(Position[:2, 3])
            # print(Position)
            ###########################
            # cv2.imshow('img1', img1)
            # cv2.imshow('img2', img2)
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()

            """start: form rigid body transformation matrix using R and T"""
            rigidTransformation_fromLast = np.concatenate((rotation, translation), axis=1)
            rigidTransformation_fromLast = np.concatenate((rigidTransformation_fromLast, np.array([[0, 0, 0, 1]])), axis=0)
            rigidTransformation_fromInitial = np.dot(rigidTransformation_fromInitial, rigidTransformation_fromLast)
            position = rigidTransformation_fromInitial[0:2, 2]
            print(position)
            """end"""
            plt.scatter(position[0], -position[1], color='r')
            plt.pause(1)

            # PPose = NPose
            PPose = NPose
        img2 = img1.copy()


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
