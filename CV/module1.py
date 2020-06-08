import cv2
import numpy as np

img = cv2.imread('2check.jpg')

# Making matrix for Erosion, dilation and morphing
kernel = np.ones((3, 3), np.uint8)
#kernel1 = np.ones((2, 2), np.uint8)

#alpha = 1
#beta = 42
#gamma = .9

#new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

#lookUpTable = np.empty((1,256), np.uint8)
#for i in range(256):
#    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

#res = cv2.LUT(img, lookUpTable)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# img = cv2.equalizeHist(img)

ret, th1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

# th1 = cv2.erode(th1, kernel, iterations=1)
th1 = cv2.dilate(th1, kernel, iterations=1)
# th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel1)

contours, _ = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    cv2.drawContours(img, contour, -1, (0, 0, 0), 1)

for b in contours:
    if cv2.contourArea(b) <50:
        continue
    # get the bounding rect
    elif cv2. contourArea(b) >100:
        x, y, w, h = cv2.boundingRect(b)
        # draw a green rectangle to visualize the bounding rect2
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

# edges = cv2.Canny(img, 100, 200)

# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.imshow('Result', img)
cv2.imwrite("result.jpg", img)
# cv2.imshow('Result', th1)
# cv2.imshow('Result', edges)
# cv2.resizeWindow('Result', 500, 500)
cv2.waitKey(0)