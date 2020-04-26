from __main__ import *
import numpy as np
import matplotlib.pyplot as plt


def ransacGo(data_x, data_y, p1x, p2x):
    datalen = len(data_x)
    data_yAvg=np.mean(data_y)
    Thresh = 20
    # print('Data length: ', int(datalen * .9))
    N = 0
    Sin = 0
    Stst = 0

    p = .95
    e = 1 - .5
    s = 3
    N = np.log(1 - p) / np.log(1 - (1 - e) ** s)

    NSin = 100000000000
    #
    print('N: ', N)
    count=0
    while True:

        rp1 = np.random.randint(datalen, size=1)
        xpr1 = data_x[rp1][0]
        ypr1 = data_y[rp1][0]

        rp2 = np.random.randint(datalen, size=1)
        xpr2 = data_x[rp2][0]
        ypr2 = data_y[rp2][0]

        # rp3 = np.random.randint(datalen, size=1)
        # xpr3 = data_x[rp3][0]
        # ypr3 = data_y[rp3][0]

        BMatrix = np.array([ypr1, ypr2])

        # print(BMatrix)
        A11 = xpr1
        A21 = xpr2
        # A31 = xpr3 ** 2

        AMatrix = np.array([
            [A11, 1],
            [A21, 1]
        ])

        # print(AMatrix)

        if np.linalg.det(AMatrix) != 0:
            a, b = np.linalg.solve(AMatrix, BMatrix)
        else:
            a = 1
            b = 1

        # print('a:',a)
        # print('b:',b)
        # print('c:',c)

        for index in range(0, datalen):
            # print('Index: ',index, '\n')
            # print(index)
            x = data_x[index]
            y = data_y[index]

            # print('X val: ', x, '\n')

            Error = np.abs((a*x + b) - y)

            if Error < Thresh:
                Stst = Stst + 1
                # print('STST:',Stst)
        # print('Error:', Error)
        count += 1
        if count%100==0:
            Thresh+=1
            print(Thresh)
        if Stst > Sin:
            # print('Number of Points within Threshold: ', Stst, '\n')
            Sin = Stst
            Stst = 0
            atst = a
            btst = b

            yeq = atst * data_x + btst


            # print('Close the plot to continue the program')
            # plt.scatter(data_x, yeq)
            # plt.scatter(data_x, data_y)
            # plt.ylabel('Y axis')
            # plt.xlabel('X axis')

            p = .95
            e = 1 - (Sin / datalen)
            s = 3
            NSin = np.log(1 - p) / np.log(1 - (1 - e) ** s)

            # print('NSin', NSin)
            if NSin < N:
                print('Aprroximation Achieved')
                # print('a,:', atst)
                # print('b,:', btst)
                plt.scatter(data_x, yeq)
                plt.scatter(data_x, data_y)
                plt.ylabel('Y axis')
                plt.xlabel('X axis')
                # plt.axis([0, 500, -75, 400])

                ypoints = (atst * data_x) + btst

                plt.plot(data_x, ypoints)
                plt.show()

                p1y = atst * p1x + btst
                p2y = atst * p2x + btst

                p1 = np.array([p1x, p1y]).T
                p2 = np.array([p2x, p2y]).T

                return p1, p2

        else:
            Stst = 0


