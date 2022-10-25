import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as img
import math

delta = 25  # tolerance
coefficients = []


def ransac(arr):  # arr here is the 2D array of center of circles detected i.e. row0=(x1, y1) ; row1=(x2, y2)
    # print(arr.shape[0])
    max_count = 0
    for idx, x in enumerate(arr):
        # print(idx, x)
        for idy in range(idx + 1, arr.shape[0]):
            y = arr[idy]
            point = (x, y)

            '''
            A, B, C are the coefficients of the line formed by the detected circles center.
            Ax + By + C = 0 --> equation of line
            A = (y1 - y2) ; B = (x2 - x1) ; C = (x1*y2 - x2*y1)
            '''

            A = point[0][1] - point[1][1]
            B = point[1][0] - point[0][0]
            C = (point[0][0] * point[1][1]) - (point[0][1] * point[1][0])

            '''
            Perpendicular distance from above line is --> |Ax1 + By1 + C|/sqrt(A^2 + B^2)
            If points are within tolerance or delta then they are considered as inliers
            '''
            inlierscount = 0
            for i in range(0, len(arr)):
                distance = abs((A * arr[i][0]) + (B * arr[i][1]) + C) / (pow((A * A) + (B * B), 0.5))
                if distance < delta:
                    inlierscount += 1

            if inlierscount > max_count:
                coefficients = [A, B, C]
                max_count = inlierscount

    for poi in arr:
        plt.scatter(poi[0], poi[1], color="green")

    A = coefficients[0]
    B = coefficients[1]
    C = coefficients[2]

    allxvalues = [poi[0] for poi in arr]

    xvalues = [min(allxvalues), max(allxvalues)]
    yvalues = [(-C - A * xvalues[0]) / B, (-C - A * xvalues[1]) / B]

    # slope = - (A / B)
    '''
    Code to calculate threshold/delta values line. Generates two parallel lines
    '''
    y_intercept_1 = C + delta * math.sqrt(A ** 2 + B ** 2)
    y_intercept_2 = C - delta * math.sqrt(A ** 2 + B ** 2)
    yvalues1 = [(-y_intercept_1 - A * xvalues[0]) / B, (-y_intercept_1 - A * xvalues[1]) / B]
    yvalues2 = [(-y_intercept_2 - A * xvalues[0]) / B, (-y_intercept_2 - A * xvalues[1]) / B]

    plt.plot(xvalues, yvalues, color="red")
    plt.plot(xvalues, yvalues1, color="blue")
    plt.plot(xvalues, yvalues2, color="blue")
    plt.title('Line Fitted using Ransac')
    print("Inliers count = ", max_count)
    plt.show()


'''
    Main -->
'''

input_img = cv2.imread('D:\line_ransac.png')
cv2.imshow("Input Image", input_img)

input_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

'''
Kernel is the window we have to blur in. Odd num coz helps in finding central pixel.
Average blur finds avg of surrounding pixels. Gaussian gives weights. In median we find median of the kernel and allot to central. 

Blur kernel size at 5 gives better output than 9. Helps to distinguish several dots which are together.
'''

input_gray = cv2.medianBlur(input_gray, 5)
# input_gray = cv2.bilateralFilter(input_gray, 9, 75, 75)  # sharp edges are preserved with noise removal hence best
# to use input_gray = cv2.GaussianBlur(input_gray, (5, 5), 0)  # reduces detail in image

edges = cv2.Canny(input_gray, 255, 125)
cv2.imshow("Canny", edges)
cv2.imshow('Blurred', input_gray)

# for Hough Circles reference --> https://theailearner.com/tag/cv2-houghcircles/
circle = cv2.HoughCircles(image=edges, method=cv2.HOUGH_GRADIENT, dp=1.5, minDist=8, param1=30, param2=30 / 3,
                          minRadius=3, maxRadius=8)
'''
I have used int32 data-type since operations performed on co-ordinates go beyond limit of int16.
'''
# 'circle' returned above is an array with elements --> H[x, y, r] where (x, y) is center of circle and r is radius

count = 0
model = []
if circle is not None:
    circle = np.int32(np.around(circle))  # np.around rounds off values to the nearest integer
    for i in circle[0, :]:
        cv2.circle(input_img, (i[0], i[1]), i[2], (0, 255, 0), 3)  # prints detected circle
        cv2.circle(input_img, (i[0], i[1]), 2, (0, 0, 255), 2)  # prints circle center
        count += 1
        model.append([i[0], i[1]])
else:
    print("\nNoneType Error!")

model_array = np.array(model)
# print(model_array.shape)

print("Detected Circles Count = ", count)

ransac(model_array)

cv2.imshow("Detected Circles", input_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
