#!/usr/bin/env python
# coding: utf-8

# In[7]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def get_steer_matrix_left_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_left_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                    using the masked left lane markings (numpy.ndarray)
    """
    
    
    steer_matrix_left_lane = np.zeros(shape)
    
    A = np.zeros((int(shape[0]),int(shape[1]/2)))   

    l = np.tril_indices(int(shape[0]), k=-4, m=int(shape[1]/2))
    A[l] = -0.7
    


    steer_matrix_left_lane[:, :int(shape[1]/2)] = np.fliplr(A)
    steer_matrix_left_lane[0:4, int(shape[1]/2):] = -0.7 

    
    return steer_matrix_left_lane


# In[13]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK


def get_steer_matrix_right_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_right_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                     using the masked right lane markings (numpy.ndarray)
    """
    
    steer_matrix_right_lane = np.zeros(shape)
    
    A = np.zeros((int(shape[0]),int(shape[1]/2)))   

    l = np.tril_indices(int(shape[0]), k=-4, m=int(shape[1]/2))
    
    A[l] = +0.3 
    steer_matrix_right_lane[:,int(shape[1]/2):] = A
    steer_matrix_right_lane[0:4,:int(shape[1]/2)] = 0.5
    
    
    return steer_matrix_right_lane


# In[28]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def detect_lane_markings(image):
    """
        Args:
            image: An image from the robot's camera in the BGR color space (numpy.ndarray)
        Return:
            left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
            right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """
    
    h, w, _ = image.shape
    

    # Convert the image to HSV for any color-based filtering
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Most of our operations will be performed on the grayscale version
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    sigma = 4
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    threshold = 30 # CHANGE ME
    mask_mag = (Gmag > threshold)
    
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)
    
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobely.shape)
    mask_right[:,0:int(np.floor(w/2))] = 0
    
    white_lower_hsv = np.array([0,0,150])    
    white_upper_hsv = np.array([180,100,255])   
    yellow_lower_hsv = np.array([15, 0, 102])
    yellow_upper_hsv = np.array([30, 255, 230])

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
    
    
    mask_left_edge = mask_left  * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right  * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white
    
    return (mask_left_edge, mask_right_edge)

