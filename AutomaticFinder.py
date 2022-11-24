import cv2
import os
import numpy as np
import statistics as stat
import scipy.signal as signal
from scipy.signal import savgol_filter
from cv2 import CV_8U
from matplotlib import pyplot as plt


'''
Throughout we use the term 'global flake' or 'global contour' to represent one graphene flake. Each global
flake or contour may be composed of regions of varying thicknesses, throughout this program, these regions are referred 
to as 'local flakes' or 'local contours'. A 'Global mask' refers to a 'global flake' that has had a mask applied to it, 
while a 'local mask' refers to a 'local flake' that has had a mask applied to it.
'''
###
# First we will do some pre-processing on the original sample image.
###

# Import Sample Img and Bkg
img = cv2.imread('Assets/5_30_4.jpg')
bkg = cv2.imread('Assets/bkg2.jpg')

# Finds average background color
average_color_row = np.average(bkg, axis=0)
average_bkg_color = np.average(average_color_row, axis=0)
# img = cv2.imread('Assets/6_6_2_1.jpg')
# bkg = cv2.imread('Assets/6_6_bkg.jpg')

def preProcess(img, bkg):
    # Crops image to exclude microscope and lens flare
    img = img[240:840, 460:1260]
    cv2.imshow('original', img)
    bkg = bkg[240:840, 460:1260]

    # Normalizes Img and Converts to Grayscale
    img_c = img.copy()
    img_c = np.divide(img_c, bkg) * 100
    img_norm = img_c.astype(np.uint8)
    img_gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
    img_gray = np.multiply(img_gray, 2)
    img_norm = np.multiply(img_norm, 2)

    # Apply mean-shift algorithm
    meanshift = cv2.pyrMeanShiftFiltering(img_norm, 10, 5)

    # Show images and histograms to visualize results
    cv2.imshow('processed', img_norm)
    cv2.imshow('meanshift', meanshift)
    hist_original = cv2.calcHist([img], [0], None, [255], [1, 256])
    plt.plot(hist_original, label='Original Image')
    hist_processed = cv2.calcHist([meanshift], [0], None, [255], [1, 256])
    plt.plot(hist_processed, label='Processed Image')
    plt.legend(loc="upper right")
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return meanshift, img_gray


def smoothHist(gray_img):
    """
    Is presented a grayscaled image of the original sample, returns a smoothed pixel intensity histogram.
    :param gray_img:
    :return smooth_hist:
    """
    # Calculate the pixel intensity histogram of the grayscaled image and apply savgol filters for smoothing
    hist_original = cv2.calcHist([gray_img], [0], None, [255], [1, 256])
    first_smoothing = savgol_filter(np.ravel(hist_original), 5, 2)
    second_smoothing = savgol_filter(first_smoothing, 5, 3)

    index = np.where(second_smoothing < 1)

    for i in index:
        second_smoothing[i] = 1
    return second_smoothing

###
# Now we will construct the functions necessary to automatically process our sample image.
###

def globalThresh(gray_img):
    """
    Thresholds the original normalized grayscale sample image to remove most of the substrate / noise.
    :param gray_img:
    :return img_thresh:
    """
    # We look to find the most prominent peak, representing the majority of the substrate
    hist = smoothHist(gray_img)
    max_index = np.argmax(hist)
    upperb = int(max_index - 10)

    # Now we simply threshold from 50 to slightly below the most prominent peak
    img_thresh = cv2.inRange(gray_img, 40, upperb)

    # Finally, we erode and dilate to clean up the thresholded image
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img_thresh, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    cv2.imshow('dilation', dilation)
    return dilation

def filterContours(contours):
    """
    Is presented a list of contours, returns a list of contours whose area is greater than 200 pixels. This filters out
    contours that may have been due to noise or undesirable global flakes.
    :param contours:
    :return filtered_contours:
    """
    filtered_contours = []

    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 200:
            filtered_contours.append(contours[i])

    return filtered_contours


def findGlobalContours(img, img_gray):
    """
    Is presented a normalized grayscale sample image, returns a list of masked images where each element in the list
    is an individual global flake.
    :param gray_img:
    :return global_masks:
    """
    ###
    # First we threshold our grayscaled sample image, then we use findContours to return a list of contours, where the
    # ith element in the list composes pixel positions representing the contour of the ith global flake.
    ###
    img_thresh = globalThresh(img_gray)
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    filtered_contours = filterContours(contours)

    ###
    # Now we filter out contours that have an area less than 200 pixels, this will remove a lot of noise and undesirable
    # flakes.
    ###
    global_masks = []
    filtered_thresh = np.zeros_like(img)

    for i in range(len(filtered_contours)):
        # Create a mask image that contains the contour filled in white
        temp = np.zeros_like(img)
        cv2.drawContours(temp, filtered_contours, i, color=(255, 255, 255), thickness=-1)
        cv2.drawContours(filtered_thresh, filtered_contours, i, color=(255, 255, 255), thickness=-1)

        # Saves pixel locations within ith global flake
        pts = np.where(temp == 255)

        # Masks original image for the ith global contour, and appends image to list global_masks
        mask = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
        masked_global = cv2.bitwise_and(img, img, mask=mask)
        global_masks.append(masked_global)

        # Shows individual global masks
        # cv2.imshow('mask', masked_global)
        # cv2.waitKey(0)

    # Shows thresholded image after smaller flakes are filtered out
    cv2.imshow('filtered thresh', filtered_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Shows masked image
    mask = cv2.cvtColor(filtered_thresh, cv2.COLOR_BGR2GRAY)
    masked_global = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow('Masked Image', masked_global)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return global_masks


def determineThresh(mask_global):
    """
    Is presented a masked image consisting of one global flake, and a list of prominent peak positions for the flake's
    pixel intensity histogram. Returns a list of images where the ith element is an image of the ith local flake within
    the global mask.
    :param mask_global:
    :return local_flakes:
    """
    thresholds = []

    # First we calculate the pixel intensity histogram of the global mask
    mask_gray = cv2.cvtColor(mask_global, cv2.COLOR_BGR2GRAY)

    # Apply Savgol Filtering
    hist = smoothHist(mask_gray)
    hist_original = cv2.calcHist([mask_gray], [0], None, [255], [1, 256])
    plt.plot(hist_original, label='Original')
    plt.plot(hist, label='Smoothed')

    # Then we determine the peaks of the histogram
    peak_pos = signal.find_peaks(hist.flatten(), prominence=30, distance=5, height=55)
    plt.plot(peak_pos[0].tolist(), [hist[i] for i in peak_pos[0]], 'ro', label='Peaks')
    hist.flatten()

    # Now we determine the threshold values surrounding each peak position, and append to list thresholds
    if len(peak_pos[0]) >= 2:
        for i in range(len(peak_pos[0])):
            if i == 0:
                below_index = 30
                above_index = ((peak_pos[0][i + 1] - peak_pos[0][i]) // 2) + peak_pos[0][i]

                plt.plot(below_index, int(hist[below_index]), 'yo')
                plt.plot(above_index, int(hist[above_index]), 'yo')

                thresholds.append(below_index)
                thresholds.append(above_index)

            elif i != len(peak_pos[0]) - 1 and i != 0:
                above_index = ((peak_pos[0][i + 1] - peak_pos[0][i]) // 2) + peak_pos[0][i]

                plt.plot(above_index, int(hist[above_index]), 'yo')

                thresholds.append(above_index)

            else:
                above_index = 250

                plt.plot(above_index, int(hist[above_index]), 'yo',  label='Thresholds')

                thresholds.append(above_index)
        plt.legend(loc="upper right")
        plt.show()
        return thresholds

    # Handles the case where a smaller homogenous global mask is being analyzed, and one or no peaks are detected
    plt.show()
    return [30, 250]


def localThresh(mask_global):
    """
    Is presented a masked global flake, returns a list where each element in the list is a local flake.
    :param mask_global:
    :return masks_local:
    """
    thresholds = determineThresh(mask_global)
    kernel = np.ones((3, 3), np.uint8)
    mask_gray = cv2.cvtColor(mask_global, cv2.COLOR_BGR2GRAY)
    masks_local = []

    for i in range(len(thresholds) - 1):
        mask = cv2.inRange(mask_gray, int(thresholds[i]), int(thresholds[i + 1]))

        mask_erosion = cv2.erode(mask, kernel, iterations=1)
        mask_dilation = cv2.dilate(mask_erosion, kernel, iterations=1)

        masks_local.append(mask_dilation)

    return masks_local

def findContrast(masks_local):
    """
    Takes in a list of contour, returns the contrast of the pixels
    """
    thickness = 0
    # Access the image pixels and create a 1D numpy array then add to list
    temp_pts = np.where(masks_local == 255)
    pixels = img[temp_pts[0], temp_pts[1]]
    avg_pixel = np.average(pixels, axis=0)
    contrast = abs(((.114 * average_bkg_color[2] + .587 * average_bkg_color[1] + .299 * average_bkg_color[0]) -
                    (.114 * avg_pixel[2] + .587 * avg_pixel[1] + .299 * avg_pixel[0])) / (
                               .114 * average_bkg_color[2] + .587 *
                               average_bkg_color[1] + .299 *
                               average_bkg_color[0]))
    if 0.02 < contrast < .35:
        return contrast, thickness == 1
    elif .35 < contrast < .65:
        return contrast, thickness == 2
    else:
        return contrast, thickness == 3


def AutomaticFinder(img, bkg):
    """
    Scratch function used to test different steps throughout the creation of this project.
    :param img:
    :return:
    """
    # Pre-process the images
    img, img_gray = preProcess(img, bkg)

    # Find suitable global flakes
    contours_global = findGlobalContours(img, img_gray)

    # Find all local flakes, calculate contrasts, and identify layers on original image
    height, width, _ = img.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    for mask_global in contours_global:
        masks_local = localThresh(mask_global)
        for mask_local in masks_local:
            contrast, thickness = findContrast(mask_local)
            contour, hierarchy = cv2.findContours(mask_local, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            x, y, w, h = cv2.boundingRect(contour[0])
            min_x, max_x = min(x, min_x), max(x + w, max_x)
            min_y, max_y = min(y, min_y), max(y + h, max_y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img

def testCase(img, bkg):
    """
    Scratch function used to test different steps throughout the creation of this project.
    :param img:
    :return:
    """
    img, img_gray = preProcess(img, bkg)
    global_contours = findGlobalContours(img, img_gray)

    thresh = localThresh(global_contours[2])



    for i, val in enumerate(thresh):
        cv2.imshow(str(i) + '', val)

    return img

testCase(img, bkg)
# cv2.imshow('final', AutomaticFinder(img, bkg))





# Show Images
# directory = r'C:\Users\eston\Documents\GitHub\WangSU22\Processed Samples'
# os.chdir(directory)
# cv2.imwrite('IdentifiedLayers(2).jpg', gray_mask)
# cv2.imshow("img w/ thresh", img_thresh)
# cv2.imshow("img w/ norm", img_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()
