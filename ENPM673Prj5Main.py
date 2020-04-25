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
    FixImagery=False
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

    if FixImagery:
        correctImages(directory, LUT)


    frameset = openVar('CorrectedFrames')

    for i, frame in enumerate(frameset):
        ###########################
        # Add Functions here

    # 1. Point correspondence
        p1 = np.array([
            [0, 1],
            [1, 1],
            [2, 5],
            [3, 5],
            [4, 1],
            [5, 1],
            [5, 5],
            [6, 5],
        ])

        p2 = p1 + 2

    # 2. Est fund Matrix w/ ransac
    #     2a. Center and scale to 8 point
    #     2b. Est Fund mat via ransac
    #     2c. Enforce rank 2 contraint

        F=computeFMatrix(p1, p2)

    # 3. Fin e matrix from F with calibration params
    #
    # 4. Decompe E into T and R
    #
    # 5. Find T and R solutions (cherality) use T and R giving largest  positive depth vals
    #
    # 6. plot pos of cam center based on rot and tans
    #
    #
    #
    #








        ###########################
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


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
