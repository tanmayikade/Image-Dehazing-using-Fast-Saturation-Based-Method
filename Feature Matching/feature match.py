import cv2

input1 = cv2.imread('D:\\Images\\image stitch\\foto1A.jpg')
input2 = cv2.imread('D:\\Images\\image stitch\\foto1B.jpg')
img1_gray = cv2.cvtColor(input1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY)

img1 = cv2.resize(img1_gray, (600, 500))
img2 = cv2.resize(img2_gray, (600, 500))

orb = cv2.ORB_create(nfeatures=800)  # max number of features to keep
kp1, des1 = orb.detectAndCompute(img1, None)  # detecting keypoints using FAST & computing descriptors using BRIEF
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # used the brute-force matcher

'''
Since we are using ORB hence I have used cv2.NORM_HAMMING as given in the documentation.

cross-check is True here to match the features in both set and hence give good results. 
It is an alternative for D.Lowe ratio in SIFT algo.
'''

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:35], None)
# first 35 matches are drawn for good visibility

# match_img = cv2.resize(match_img, (1100, 600))

cv2.imshow('Original 1', img1)
cv2.imshow('Original 2', img2)
cv2.imshow('Matches', match_img)

signal = cv2.imwrite('D:\Task 2 - Line Fit\Feature Match\Match-5.png', match_img)
if signal:
    print("Matched Image Saved Successfully!")

cv2.waitKey()
