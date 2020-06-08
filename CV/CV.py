import cv2
import numpy as np

#img = cv2.imread('./data/DSCN9538.jpg',0)
#img2 = cv2.imread('./data/DSCN9516.jpg',0)
#img3 = img - img2
#ret, thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)
#Take one block and subtract it with the entire image to see if you get an anomally
def cut_img2():
    img = cv2.imread('2.jpg',0)
    blur = cv2.GaussianBlur(img,(11,11),0)
    edges = cv2.Canny(blur, 9,26)
    h,w = img.shape

    for row in range(0 ,h-10):
        for col in range (0, w-10): 
            dome = edges[row:row+10,col:col+10]
            #cv2.circle(edges, (row,col), 4, 255,4)
        
            if ( np.sum(dome) == 0):
                #print(np.sum(dome))
                cv2.circle(edges, (col,row), 4, 255,4)

    cv2.imshow('image', edges)

    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    cv2.imwrite("something.jpg", edges) 
    cv2.waitKey(0)


def block_scaning4():
    img = cv2.imread('4.jpg',0)
    #block1 = img[300:320, 300:320]
    h,w = img.shape
    for row in range(0,h - 20):
        for col in range(0,w-20):
            if (np.sum(img[300:320, 300:320]) -np.sum(img[row:row+20,col:col+20]) != 0):
                print("fookin anomaly")

    cv2.imshow('image', img)

    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600,600)
    cv2.imwrite("something2.jpg", img) 
    cv2.waitKey(0)


#block_scaning4()

#img = cv2.imread('2.jpg')

## Making matrix for Erosion, dilation and morphing
#kernel = np.ones((3, 3), np.uint8)
#kernel1 = np.ones((1, 2), np.uint8)


#img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

## img = cv2.equalizeHist(img)

#ret, th1 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

#th1 = cv2.dilate(th1, kernel, iterations=1)
## th1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel)


## edges = cv2.Canny(img, 100, 200)

## cv2.imshow('Result', img)
#cv2.imshow('Result', th1)
#cv2.imwrite("result.jpg", th1)
## cv2.imshow('Result', edges)
## cv2.resizeWindow('Result', 500, 500)
#cv2.waitKey(0)
#end copy


img = cv2.imread('6.jpg')

# Making matrix for Erosion, dilation and morphing
kernel = np.ones((3, 3), np.uint8)
kernel1 = np.ones((2, 2), np.uint8)

img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

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
    elif cv2.contourArea(b) >150:
        x, y, w, h = cv2.boundingRect(b)
        # draw a green rectangle to visualize the bounding rect2
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

# edges = cv2.Canny(img, 100, 200)

cv2.imshow('Result', img)
cv2.imwrite('Result.jpg', img)
# cv2.imshow('Result', th1)
# cv2.imshow('Result', edges)
# cv2.resizeWindow('Result', 500, 500)
cv2.waitKey(0)

#End copy


#dilate= cv2.dilate(edges, (5,5)) 
#contours, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#if cv2.contourArea(contours) > 540:
#   dilate = cv2.drawContours(dilate, [box], 0, (0,255,0), 3)

#for c in contours:     
#    rect = cv2.minAreaRect(c)
#    box = cv2.boxPoints(rect) 
#    box = np.int0(box)
#    img = cv2.drawContours(thresh, [box], 0, (0,255,0), 3)


#for c in contours: 
#    rect = cv2.minAreaRect(c)
#    box = cv2.boxPoints(rect) 
#    box = np.int0(box)
#    if cv2.contourArea(c) > 540:
#        thresh = cv2.drawContours(thresh, [box], 0, (0,255,0), 3)
#for row in range(0,h-10):    
#    for col in range (0,w-10):
#        if np.sum(thresh[row:row+10,col:col+10])== 255:
#            print("lul")
#            cv2.circle(thresh, (col,row), 4, 0,4)


#edges = cv2.Canny(img, 0, 255)
#kernel = np.ones((1,2))
#erosion = cv2.erode(edges, kernel, iterations = 1)

#contour, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#for c in contour:
#    #print (c)
#    rect = cv2.minAreaRect(c)
#    box = cv2.boxPoints(rect) 
#    box = np.int0(box)
#    edges = cv2.drawContours(edges, [box], 0, (0,255,0), 3)

#img2 = cv2.imread('something.jpg',0)
#edges2 = cv2.Canny(img, 50, 50)
#kernel2 = np.ones((2,2))
#erosion2 = cv2.erode(edges2, kernel2, iterations = 1)

#ew = erosion - erosion2
#img = cv2.imread('1.jpg')
#imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(imgray, 180, 255, 0)
#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#for c in contours:     
#    rect = cv2.minAreaRect(c)
#    box = cv2.boxPoints(rect) 
#    box = np.int0(box)
#    img = cv2.drawContours(thresh, [box], 0, (0,255,0), 3)

#cv2.imshow('image', img)
