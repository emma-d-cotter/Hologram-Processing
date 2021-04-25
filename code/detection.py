import time
import numpy as np
import torch
from utils import *
from params import *
from reconstruction import *
import scipy
import cv2
import skimage.measure
from itertools import compress

def lpf_detection(holo,mask,erode_size=20, dilate_size=60, threshold=10, A_min = 1, show_plot=False):
    # mask for low pass filter generated by create_lpf_mask (calculated outside of this function to
    #     optimize processing time)
    # erode size and dilate size are in number of pixels
    # threshold is in raw intensity
    # A_min is in mm^2


    # calculate image FFT
    #img = np.float32(holo.squeeze())
    dft = np.fft.fft2(holo)
    dft_shift = np.fft.fftshift(dft)
    f_ishift = np.fft.ifftshift(dft_shift)

    # apply mask and calculate inverse FFT
    fshift = dft_shift * mask

    proc_im = np.fft.ifft2(fshift)
    proc_im = np.abs(proc_im)

    # apply threshold to create binary image
    proc_im[proc_im < threshold] = 0
    proc_im[proc_im > 0] = 1

    # perform erosion and dilation
    if erode_size > 0:
        eroder = cv2.getStructuringElement(cv2.MORPH_RECT,(erode_size,erode_size));
        proc_im = cv2.morphologyEx(proc_im, cv2.MORPH_ERODE, eroder)

    if dilate_size > 0:
        dilater = cv2.getStructuringElement(cv2.MORPH_RECT,(dilate_size,dilate_size));
        proc_im = cv2.morphologyEx(proc_im, cv2.MORPH_DILATE, dilater)

    # find regions in binary image
    labeled_im = skimage.measure.label(proc_im,connectivity=2)
    regions = skimage.measure.regionprops(labeled_im)
    A = np.array([region.area for region in regions])
    regions = list(compress(regions,A > A_min/((pix_size*1e3)**2)))

    if show_plot:
        plt.figure(4).clf()
        plt.subplot(212)
        plt.imshow(proc_im)

        plt.subplot(211)
        plt.imshow(-holo)

        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            plt.plot(bx,by,'-r',linewidth=2)

    return regions

def create_lpf_mask(holo,keep_frac=0.05):
    # create low pass filter mask, keeping only "keep_frac"
    # fraction of lowest frequencies
    # calculating this once for all holos saves about 0.04 seconds per hologrma

    # e.g., keep_frac = 0.05 keeps 5% of frequencies

    rows, cols = holo.shape
    crow, ccol = int(rows / 2), int(cols / 2)

    mask = np.zeros((rows, cols), np.uint8)
    rr = keep_frac*(rows/2)
    rc = keep_frac*(cols/2)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2/rr**2 + (y - center[1]) ** 2/rc**2 <= 1
    mask[mask_area] = 1

    return mask
