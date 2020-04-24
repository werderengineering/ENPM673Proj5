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
    FixImagery=True
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
