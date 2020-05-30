# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:11:08 2019

@author: Mohammadreza
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


#img = plt.imread('f.png', format=None)
img = cv2.imread('f3.png', cv2.IMREAD_GRAYSCALE)

#plt.imshow(img)
#plt.figure()

res, img = cv2.threshold(img,127,1,cv2.THRESH_BINARY)
img = 1 - img

img = np.array([[1,0,1],[0,1,0],[0,1,1]])


plt.imshow(img,cmap='gray')
plt.figure()
#print(img.tolist())
height = img.shape[0]
width = img.shape[1]


vert_img = np.sum(img,axis=0)
hor_img = np.sum(img,axis=1)

#print(len(vert_img))
#print(len(hor_img))
vert_img_sort = vert_img#sorted(vert_img, key=int, reverse=True)
hor_img_sort = hor_img#sorted(hor_img, key=int, reverse=True)


#print(hor_img_sort)
#print(vert_img_sort)

def check():
    for row in hor_img_sort:
        if row != 0:
            return False
    for col in vert_img_sort:
        if col != 0:
            return False
    return True


def recursive(vert_img_sort, hor_img_sort):

    new_img = np.zeros((height,width), np.uint8)
    #height1 = new_img.shape[0]
    #width1 = new_img.shape[1]
    #print(height1)
   #print(width1)
    if (check()):
        print("Solved")
        return True

    #for row1 in range(width1):
        #for col1 in range(height1):
    i = 0
    for row in new_img:
        j = 0
        for col in new_img:
            if hor_img_sort[i] != 0 or vert_img_sort[j] != 0 and new_img[i][j] != 1:
                new_img[i][j] = 1
                hor_img_sort[i] -= 1
                vert_img_sort[j] -= 1
               # print("place")
                    
                if (recursive(vert_img_sort,hor_img_sort)):
                    return True
                
                new_img[i][j] = 0
                hor_img_sort[i] += 1
                vert_img_sort[j] +=1
                    
                   # return recursive(vert_img_sort, hor_img_sort)
                    #break
            j += 1
            i += 1
                
    #print(hor_img_sort)
    plt.imshow(new_img,cmap='gray')
    plt.figure()

new_img = np.zeros((height,width), np.uint8)
recursive(vert_img_sort, hor_img_sort)
