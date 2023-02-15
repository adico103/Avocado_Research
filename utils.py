import cv2 as cv
import numpy as np
import random as rng
import pandas as pd
import utils
import os

    # def extract_by_band(self,band):

    #     for avocado_num,box in zip(self.avocado_num,self.bounding_boxes):
    #         k = np.zeros_like(self.avocado_pixs)
    #         k[self.avocado_pixs==avocado_num] = self.spec_img[:,:,band][self.avocado_pixs==avocado_num]
    #         k = k[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]

    #     return k

def ndvi(images):
    for i in range(len(images)):
        image = scale_img(images[i])
        images[i]= image

    img_ndvi = np.zeros_like(images[0])
    img_ndvi = cv.subtract(images[1],images[0],img_ndvi)
    return img_ndvi

def norm_img(image):
    min_val = image.min()
    max_val = image.max()
    norm_image = (image - min_val) / (max_val - min_val)
    scaled_image = (norm_image * 255).astype(np.uint8)
    return scaled_image



def scale_img(image):
    # min_val = image.min()
    # max_val = image.max()
    # norm_image = (image - min_val) / (max_val - min_val)
    # scaled_image = (norm_image * 255).astype(np.uint8)
    scaled_image = (image/(2^4)).astype(np.uint8)
    return scaled_image

def get_contours(img_filter,draw=False):
  
    ret, thresh = cv.threshold(img_filter, 10, 255, cv.THRESH_BINARY)

    try:
        contours, _ = cv.findContours(image=thresh, mode=cv.RETR_CCOMP, method=cv.CHAIN_APPROX_SIMPLE)
    except ValueError:
        _, contours, _ = cv.findContours(image=thresh, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
    
    if len(contours)>0:
        idx = 0
        contours_lenght = 0

        for i, c in enumerate(contours):
            if contours_lenght < len(c):
                contours_lenght = len(c)

        img_filter = cv.cvtColor(img_filter,cv.COLOR_GRAY2BGR)

    else:
        print('no contours found')
        contours = 0

    return img_filter, contours

def draw_segmentation(centers,avocado_nums,contours,image,name='bounding_box'):
    image = cv.cvtColor(image,cv.COLOR_GRAY2BGR)
    for i in range(len(contours)):
        x,y,w,h = cv.boundingRect(contours[i])
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    for center,avocado_num in zip(centers,avocado_nums):
        cv.putText(image,str(avocado_num),center.astype(int),cv.FONT_HERSHEY_COMPLEX, 3,
                        (245,117,10),5)
    cv.imwrite(name+'.png',image)
        
        


def avocados_from_contours(image,contours):
    avocado_pixels = np.zeros_like(image)
    
    largest_contours = sorted(contours, key=cv.contourArea, reverse=True)
    i=0
    selected_countours = []
    all_found = False
    bounding_boxs = []
    while not all_found:
        cont_size = cv.contourArea(largest_contours[i])
        if cont_size>20000:
            bounding_boxs.append(cv.boundingRect(largest_contours[i]))
            selected_countours.append(largest_contours[i])
            

        else:
            all_found=True
        i+=1

    return bounding_boxs,selected_countours

def segmentation(images):
    
    nvdi_img = ndvi(images)
    _,contours = get_contours(nvdi_img)
    bounding_boxs,selected_countours = avocados_from_contours(nvdi_img,contours)

    return bounding_boxs,selected_countours,nvdi_img

def get_avocado_pix(avocado_num,contours,image):
    # image = cv.cvtColor(image,cv.COLOR_GRAY2BGR)
    all_pixs = []
    for i in range(len(contours)):
        # new_im = np.zeros_like(image)
        cv.drawContours(image=image, contours=contours, contourIdx=i, color=avocado_num[i], thickness=-1)
        # all_pixs.append(new_im)
    return image

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result
