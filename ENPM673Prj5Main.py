import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
import os
import sys
import random
import pickle
import time
import datetime
from sklearn import linear_model, datasets

from Oxford_dataset.ReadCameraModel import *
from Oxford_dataset.UndistortImage import *
from imageCorrection import *
from fundamentalMatrix import *
from featureMatch import *
from findCameraPose import *
from comparable import *
from Vizualization import *

# from useBuiltinFunction import *
from test3 import *

print('Imports Complete')

print('CV2 version')
print(cv2.__version__)

flag = True
prgRun = True
customFun_MatchFeat = True
customFun_fundamentalMatrix = False
customFun_cameraPose = False
customFun_Homo = True
Saveoff = 'Allvisited points'

JustViz = False


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
    #     'What is the name and directory of the folder with the images? Note, this should be entered as"./folder_name if on Windows". Example: ./Oxford_dataset/stereo/centre/: \n'))

    # JustVizYN = str.lower(input('\nJust the visual? Enter Yes if you just want to see the last saved visual\n'))
    #
    # if JustVizYN == 'yes':
    #     JustViz = True
    # else:
    #     print('Running Stereo Process')
    #     JustViz = False

    # OpenCVYN = str.lower(input('\nUse OpenCV? Enter Yes if you just use openCV functions\n'))
    # if OpenCVYN == 'yes':
    #     customFun_MatchFeat = False
    #     customFun_fundamentalMatrix = False
    #     customFun_cameraPose = False
    #     customFun_Homo = False
    #     Saveoff = 'CV2Comparison'
    # else:
    #     customFun_MatchFeat = True
    #     customFun_fundamentalMatrix = True
    #     customFun_cameraPose = True
    #     customFun_Homo = True
    #     Saveoff = 'Allvisited points'

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

    EndProgram = False

    if FixImagery:
        correctImages(directory, LUT)
    try:
        print('\nUnpickling video data')
        frameset = openVar('CorrectedFrames')
        print('\nVideo Unpickled')
    except:
        print("MISSING FRAMES\nPlease re-run and create the data set")
        EndProgram = True

    if JustViz == False and EndProgram == False:
        startTime = time.time()
        print('Generating points. This will take a while')

        for i, frame in enumerate(frameset):
            ###########################
            # Add Functions here
            imgCur = frame.copy()

            ######
            # Change this value to make th test longer or shorter. Good comparison is 500
            NFrames = 100
            # NFrames=len(frameset)
            if i > NFrames:
                break
            if i > 19:

                if i % 10 == 0 or i == 20:
                    if i == 20:
                        Pframes = 20
                        pEpochT = startTime
                    else:
                        Pframes = i
                        pEpochT = epochTime

                    epochTime = time.time()
                    print('\nEpoch Time (s): ', str(datetime.timedelta(seconds=epochTime - pEpochT)))
                    print('Total Time: ', str(datetime.timedelta(seconds=epochTime - startTime)))
                    print('Epoch: ', i)

                    if i != 20:
                        TotalTime = time.time() - startTime
                        RPF = (TotalTime / i) * NFrames
                        print('Estimated time to complete all desired frames: ', str(datetime.timedelta(seconds=RPF)))
                        print('Estimated time remaining: ', str(datetime.timedelta(seconds=RPF - TotalTime)))
                # 1. Point correspondence

                # print(imgPrev.shape)
                if customFun_MatchFeat:

                    PointsPrev, PointsCur = matchUntil(imgPrev, imgCur, sift)
                else:
                    PointsPrev, PointsCur = getcomparablePoints(imgPrev, imgCur, sift)

                # 2. Est fund Matrix w/ ransac
                #     2a. Center and scale to 8 point
                #     2b. Est Fund mat via ransac
                #     2c. Enforce rank 2 contraint

                if customFun_fundamentalMatrix:
                    F, PointsPrev, PointsCur = computeFMatrix(PointsPrev, PointsCur)

                else:
                    F, PointsPrev, PointsCur = getcomparableF(PointsPrev, PointsCur)

                if customFun_cameraPose:
                    # 3. Fin e matrix from F with calibration params
                    essentialMatrix = essentialMatrixFromFundamentalMatrix(F, K)
                    # 4. Decompe E into T and R
                    rotations_fromLastToCurrent, translations_fromLastToCurrent = extractCameraPoseFromEssntialMatrix(
                        essentialMatrix)
                    # 5. Find T and R solutions (cherality) use T and R giving largest  positive depth vals
                    # if customFun_cameraPose:
                    rotation_fromLastToCurrent, translation_fromLastToCurrent, votes, index_maxVote = DisambiguateCameraPose(
                        rotations_fromLastToCurrent, translations_fromLastToCurrent, K,
                        np.float32(PointsCur),
                        np.float32(PointsPrev))
                else:
                    rotation_fromLastToCurrent, translation_fromLastToCurrent = getcomparableRT(F, K, PointsCur,
                                                                                                PointsPrev)

                    # rotation,translation=poseMatrix(imgPrev,imgCur,K,sift)

                # 6. plot pos of cam center based on rot and tans
                """start: form rigid body transformation matrix using R and T"""
                if customFun_Homo:
                    rigidTransformation_fromLast = homoTransformation(rotation_fromLastToCurrent,
                                                                      translation_fromLastToCurrent)
                    rigidTransformation_fromInitial = rigidTransformation_fromInitial @ rigidTransformation_fromLast
                    PointsPrev = np.dot(rigidTransformation_fromInitial, pose_initial)
                else:
                    homo2 = getcomprableHomo(rotation_fromLastToCurrent, translation_fromLastToCurrent)
                    homo1 = np.dot(homo1, homo2)
                    PointsPrev = np.dot(homo1, t1)
                """end"""

                listPose.append([PointsPrev[0][0], -PointsPrev[2][0]])

            imgPrev = imgCur.copy()

        TotalTime = time.time() - startTime
        TPF = (TotalTime / i) * len(frameset)
        print('Estimated time to complete all frames: ', str(datetime.timedelta(seconds=TPF)))
        saveVar(listPose, Saveoff)

    if EndProgram == False:
        # playViz('Allvisited points','CV2Comparison', frameset)
        showTracking('Allvisited points', 'CV2Comparison')

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
