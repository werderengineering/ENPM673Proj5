import cv2
import os
import numpy as np
# for filename in os.listdir('./Oxford_dataset/stereo/centre/'):
#     print(filename)

img1 = cv2.imread('./Oxford_dataset/stereo/centre/1399381446392174.png')
img2 = cv2.imread('./Oxford_dataset/stereo/centre/1399381446454665.png')

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# # Brute force Matcher
# # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des1, des2)
# matches = sorted(matches, key=lambda x: x.distance)
# match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
# cv2.imshow('Matches', match_img)
# cv2.waitKey()
############################################
# FLANN matcher
index_params = dict(algorithm=6,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=2)
search_params = {}
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None)

#Select points to match
features = []
for m, n in matches:
    if m.distance < 0.5 * n.distance:
        features.append([m])
match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, features, None, flags=2)


imgs = np.vstack([knn_image, match_img])
cv2.namedWindow("Matches",0);
cv2.resizeWindow("Matches", 1000, 600);
cv2.imshow('Matches', imgs)
cv2.waitKey(0)
