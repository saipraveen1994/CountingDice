# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 19:51:21 2018

@author: Saipraveen Vabbilisetty
"""
import cv2
import numpy as np
import sys
argList = sys.argv
#############################################################################
# A function to count the number of pips on dice
def main():
        img = cv2.imread(argList[1],0)
        # Reading Image from the user and converting to a grayscale one
        th,dst = cv2.threshold(img,150,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #cv2.imshow('threshold',img) Uncomment this line to see the image after applying threshold
        img = cv2.Canny(dst,127,255)
        #cv2.imshow('Canny',img) Uncomment this line to see the image after applying Canny filter
        mask = np.ones((3,3),np.uint8)
        img = cv2.dilate(img,mask,iterations=1)
        #cv2.imshow("Amplification",img) Uncomment this line to see the image after dilation (Dilation is to amplify the quality of objects)
        (_, cnts, _) = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
        # Finding Contours and sorting them based on area (top 10)
        # loop over our contours
        for c in cnts:
                peri = cv2.arcLength(c, True)
                #Since the dice resembles a rectangle/square taking the perimeter and approximating it
                approx = cv2.approxPolyDP(c, 0.01 * peri, True)
                area = cv2.contourArea(c)
                flag = 0
                # Filtering the regions based on area (to eliminate noise regions if any)
                if area >5000 and area < 25000:
                        detector = cv2.SimpleBlobDetector_create()
                        # Blob detector to detect pipes
                        keypoints = detector.detect(img)
                        for point in keypoints:
                            diameter = point.size
                            # Filtering the pipes based on diameter (On Observation the daimeter of pipes is > 10)
                            if diameter >10.0:
                                    flag = 1
                            else:
                                    flag = 0
                        if flag > 0:
                            im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]),(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                            #cv2.imshow("Keypoints", im_with_keypoints) Uncomment this line to see the image after identifying pipes
                            count = len(keypoints)
                            # Counting number of pipes based on identified circles
                            resultText = "Sum : " + " -->" + " " + str(count)
                            # Building Result String
                            writeResultOnImage(im_with_keypoints, resultText)
                            # Calling the function to write text on Image
                            cv2.imshow('Output',im_with_keypoints)
                            break        
                        else:
                            count = 0
                            resultText = "No die identified" + " " + "Sum : " + " -->" + " " + str(count)
                            image = cv2.imread(argList[1])
                            writeResultOnImage(image, resultText)
                            cv2.imshow('Output',image)
                            break
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#######################################################################################################################
def writeResultOnImage(openCVImage, resultText):
    # ToDo: this function may take some further fine-tuning to show the text well given any possible image size

    imageHeight, imageWidth, sceneNumChannels = openCVImage.shape

    # choose a font
    fontFace = cv2.FONT_HERSHEY_TRIPLEX

    # chose the font size and thickness as a fraction of the image size
    fontScale = 1.0
    fontThickness = 2

    # make sure font thickness is an integer, if not, the OpenCV functions that use this may crash
    fontThickness = int(fontThickness)

    upperLeftTextOriginX = int(imageWidth * 0.05)
    upperLeftTextOriginY = int(imageHeight * 0.05)

    textSize, baseline = cv2.getTextSize(resultText, fontFace, fontScale, fontThickness)
    textSizeWidth, textSizeHeight = textSize

    # calculate the lower left origin of the text area based on the text area center, width, and height
    lowerLeftTextOriginX = upperLeftTextOriginX
    lowerLeftTextOriginY = upperLeftTextOriginY + textSizeHeight

    # write the text on the image
    cv2.putText(openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, (255,255.255) , fontThickness)
# end function

#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    main()
