import pandas
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import os
import glob
from scipy import stats
import torch
import time
from params import *
import warnings

# basic utilities for hologram processing
# Emma Cotter 2021

def norm01(x):
    # normalize a vector between 0 and 1
    xnorm = (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))
    return xnorm

def extract_region(holo,bbox,pad_size):
    # extract region from a hologram with pad_size mm added around the bbox
    # will terminate at edges of hologram (no zero-padding)
    # inputs:
    # holo - full hologram
    # bbox - bounding box of roi
    # pad_size - amount of padding to add to each side of the bbox (see figure in pub)
    pix_mm = 1e-3/pix_size
    reconstruct_pad = pad_size*pix_mm
    minro, minco, maxro, maxco = bbox

    if (minro - reconstruct_pad) >= 0:
        minr = int(minro - reconstruct_pad)
        xmin = int(reconstruct_pad)
    else:
        minr = 0
        xmin = int(reconstruct_pad - np.abs(minro - reconstruct_pad))

    # bottom column inxex
    if (minco - reconstruct_pad) >= 0:
        minc = int(minco - reconstruct_pad)
        ymin = int(reconstruct_pad)
    else:
        minc = 0
        ymin = int(reconstruct_pad - np.abs(minco - reconstruct_pad))

    # max row index
    if (maxro + reconstruct_pad) <= holo.shape[0]:
        maxr = int(maxro + reconstruct_pad)
        xmax = int((maxr - minr) - reconstruct_pad)
    else:
        maxr = int(holo.shape[0])
        xmax =  int(xmin + (maxro - minro))

    # max column index
    if (maxco + reconstruct_pad) <= holo.shape[1]:
        maxc = int(maxco + reconstruct_pad)
        ymax = int((maxc - minc) - reconstruct_pad)
    else:
        maxc = int(holo.shape[1])
        ymax =  int(ymin + (maxco -minco))


    if len(holo.shape) == 3:
        holoreg = holo[minr:maxr,minc:maxc,:]
    else:
        holoreg = holo[minr:maxr,minc:maxc]
    reginds = (xmin,ymin,xmax,ymax)
    oginds = (minr,minc,maxr,maxc)

    return holoreg, reginds, oginds

def complexto2D(X, device):
    # spectral tensor operations can't operate on complex tensors. a + bi must be [a b]
    # for a 2D tensor, this adds a third dimension so that one plane
    # has real and one plane has imaginary components
    # tensor is on specified device (CPU or CUDA)
    s = X.shape
    X2 = torch.zeros((s[0],s[1],2)).float().to(device)
    X2[:,:,0] = torch.real(X)
    X2[:,:,1] = torch.imag(X)
    return X2

def tocomplex(X, device):
    # inverse of complexto2D - back to a complex tensor
    s = X.shape
    X2 = torch.zeros((s[0],s[1])).type(torch.cfloat).to(device)
    X2 = X[:,:,0] + 1j*X[:,:,1]
    return X2

class HoloDisplay:
    # basic hologram disply. right click to get farther, left click to get closer
    # starts display at index = idx0
    def __init__(self,R,z,idx0):
        self.idx = idx0

        fig = plt.figure(1)
        fig.clf()
        img = plt.imshow(R[:,:,self.idx],cmap=plt.get_cmap('pink'))

        plt.title(z[self.idx])
        plt.colorbar()
        plt.draw()
        self.dd = img
        self.z = z
        self.R = R
        self.cid = img.figure.canvas.mpl_connect('button_press_event',self)

    def __call__(self,event):
        if event.button == 1:

            self.idx += 1
            if self.idx > (len(self.z)-1):
                self.idx -= 1

        elif event.button == 3:
            self.idx -= 1
            if self.idx < 0:
                self.idx += 1

        self.dd.set_array(self.R[:,:,self.idx])
        plt.title(str(self.idx) + ': ' + str(self.z[self.idx]))
        plt.draw()
