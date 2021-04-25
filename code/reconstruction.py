import time
import numpy as np
import torch
import cv2
from utils import *
from params import *
from detection import *
import warnings
# written by: Emma Cotter
# emma.d.cotter@gmail.com

# functions for reconstruction of holograms
# includes:
# physical_to_optical and optical_to_physical - functions to convert between physical 
#       and optical pathlength given indices of refraction of water and air
# reconstruct - reconstruct a hologram at specified plane(s)
# holofft and propogate - worker functions for reconstruction
# bg_subtract - perform background subtraction (should be done before reconstruction)
# gsfocus - perform golden section search optimized auto-focus
# stdcorr - calculate standard deviation correlation focus metric

def physical_to_optical(L):
    # convert from physical to optical path length
    # (e.g., if physical distance is 500 mm from camera, what is the propgation
    # distance z to use in reconstruction?)
    if L < L_air:
        L_optical = L_air/n_air
    elif L <= (L_air + L_water):
        L_optical = L_air/n_air + (L-L_air)/n_water
    else:
        warnings.warn('Warning! Requested path length is longer than separation distance')
        L_optical = L_air/n_air + (L-L_air)/n_water

    return L_optical

def optical_to_physical(L):
    # convert optical path length to phytical path length (e.g., if auto-focus
    # says that target is 500 mm from the camera, how far is it *really*?)
    if L < L_air*n_air:
        L_physical = L*n_air
    elif L <= (L_air*n_air + (L_water-L_air)*n_water):
        L_physical = L_air*n_air + (L-L_air)*n_water
    else:
        warnings.warn('Warning! Requested path length is longer than separation distance')
        L_physical = L_air*n_air + (L-L_air)*n_water
    return L_physical

def reconstruct(holo,zstack,useGPU=True,savedata=False,outdir=None,outputdata=True):
    # function for hologram resconstruction
    # performs all eperations as pytorch tensors,
    # so that it can be performed on GPU
    # input is PHYSICAL distance, not OPTICAL distance

    # inputs:
    # holo - background-subtracted hologram or hologram region (numpy array)
    # zstack - reconstruction plane distance or distancess (physical path length, not optical)
    # useGPU - use GPU if available (boolean)
    # savedata - save reconstructions to disk? (boolean)
    # outdir - path to store reconstructions, if requested (str)
    # outputdata - return reconstructions? It can be helpful to set this to False
    #               if many reconstructions are being calculated and saved to disk,
    #               but you don't want to keep them in memory

    # returns: rstack - reconstruction or reconstructions at requested planes
    # ~*~*~*~*~*~*~*~*~*~*~*~*~*~*~

    # create directory if output is requested
    if (savedata) and (outdir is None):
        raise "Must specify out directory to store data"
    if (savedata) and (not os.path.isdir(outdir)):
        os.mkdir(outdir)

    # start timer
    t = time.time()

    # calculate FFT of Holograms (only do this once per hologram, regardless of
    # hot many reconstructions are requested)
    A,params = holofft(holo,useGPU)

    # indexing is different if single reconstrunction plane or multiple
    if type(zstack) is int or type(zstack) is float:
        z = zstack
        rstack = propogate(physical_to_optical(z), A, params, useGPU)

    else:

        if outputdata:
            rstack = torch.zeros((holo.shape[0],holo.shape[1],len(zstack)))
        else:
            rstack = None
        for i, z in enumerate(zstack):
            im = propogate(physical_to_optical(z),A,params,useGPU)
            if savedata:
                zstr = str(np.round(optical_to_physical(z)*10000)).zfill(5)
                outfile = os.path.join(outdir,zstr+'.jpg')
                cv2.imwrite(outfile,im.cpu().numpy())
            if outputdata:
                rstack[:,:,i] = im


    elapsed = time.time() - t
    #print('Reconstruction took: %s' % (elapsed))

    return rstack, elapsed


def holofft(holo,useGPU=True):
    # first step in reconstruction (so that the fft does not need
    # to be calculated multiple times for auto-focusing)

    # input : hologram or hologram region
    # output: 2d FFT and relevant values for propogation

    H = holo

    # get dimensions of hologram
    No = H.shape[0]
    Mo = H.shape[1]

    # zero padding based on hologram dimensions
    if Mo < No:

        d = int(np.ceil((No-Mo)/2))
        if (d-((No-Mo)/2)) != 0:
            H = np.pad(H,((0,1),(d,d)))
            N = No + 1
            M = No + 1
            p1 = 1 # add one to pad
        else:
            H = np.pad(H,((0,0),(d,d)))
            N = No
            M = No
            p1 = 0 # add zero to pad
        ar = 0 # tall image

    elif No < Mo:
        d = int(np.ceil((Mo-No)/2))
        if (d-((Mo-No)/2)) != 0:
            H = np.pad(H,((d,d),(0,1)))
            N = Mo + 1
            M = Mo + 1
            p1 = 1 # add one to pad
        else:
            H = np.pad(H,((d,d),(0,0)))
            N = Mo
            M = Mo
            p1 = 0 # add zero to pad
        ar = 1 # squat image
    else:
        ar = 0
        d = 0
        N = No
        M = Mo
        p1 = 0;

    # convert to tensor and send to GPU if available/requested
    if useGPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    H = torch.from_numpy(H).type(torch.cfloat).to(device)

    # create grid of indices
    u = torch.arange(1,N+1)
    v = torch.arange(1,M+1)
    V,U = torch.meshgrid(v,u)
    L = pix_size*N # side length
    alpha = wavelength*(U-N/2-1)/L
    beta = wavelength*(V-M/2-1)/L

    # terms needed for reconstruction
    f1 = torch.exp(1j*np.pi*(U+V)).to(device)
    f2 = torch.exp(-1j*np.pi*(U+V)).to(device)
    alpha = wavelength*(U-N/2-1)/L
    beta = wavelength*(V-M/2-1)/L

    # calculate FFT of hologram
    A = complexto2D(f1*H,device)
    A = torch.fft(A,2)
    A = tocomplex(A, device)

    # store parameters needed for propogation
    params = {
        'M':M,
        'N':N,
        'd':d,
        'beta':beta,
        'alpha':alpha,
        'p1':p1,
        'ar':ar,
        'f1':f1,
        'f2':f2
    }

    return A, params

def propogate(z, A, params,useGPU=True):
    # propogate hologram to distance z from sensor (perform reconstruction)
    # inputs
    # z - distance, in m
    # A, params - output from "holofft"

    # extract params
    M = params['M']
    N = params['N']
    d = params['d']
    beta = params['beta']
    alpha = params['alpha']
    p1 = params['p1']
    ar = params['ar']
    f1 = params['f1']
    f2 = params['f2']

    # convert to tensor and send to GPU if available/requested
    if useGPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    # calculate propogater
    S = torch.exp(-(2*np.pi*z*1j/wavelength)*
                  torch.sqrt(1-torch.square(alpha)-torch.square(beta))).to(device)

    # propogate the image spectrum
    step = (f1*A)*S

    # inverse FFT
    step = complexto2D(f2*step, device)
    step = torch.ifft(step,2)
    step = tocomplex(step, device)
    step = torch.abs(f2*step)

    # normalize
    m = torch.max(step)
    step = 255*step/m

    # trim back to original dimensions
    if ar == 1:
        if p1 == 1:
            img = step[d:M-d,0:-1]
        else:
            img = step[d:M-d,:]
    else:
        if p1 == 1:
            img = step[0:-1,d:M-d]
        else:
            img = step[:,d:M-d]

    return img

# background subtraction function
def bg_subtract(holoname, holometa):
    # holoname: name of hologram file (excluding .tif)
    # holometa: hologaphic camera metadata structure for a dive (ouput of load_holometa)
    #           (pandas data frame with paths and timestamps of all recorded holograms)
    file_idx = int(np.where([[holoname in path] for path in holometa['file']])[0])
    imagepath = holometa['file'][file_idx]
    fileroot = imagepath.split('\\')[-1][:-4]
    holo = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)

    # find 5 previous files for background
    bg_idx = np.arange(file_idx-5,file_idx)

    # check that 5 previous files are at approximately the same depth (within 3 m).
    # Don't process if there are fewer than 3 background files
    bg_idx = bg_idx[np.abs(holometa['z'][file_idx] - holometa['z'][bg_idx]) < 3]
    if bg_idx.size > 2:
        bg = np.zeros((holo.shape[0],holo.shape[1],bg_idx.shape[0]))
        for i,idx in enumerate(bg_idx):
            imagepath = holometa['file'][idx]
            bgim =cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
            bg[:,:,i] = bgim

        bg = np.median(bg,axis=2)
        holo = holo - bg
        flag = 0
    else:
        print('Sufficient Background data not available')
        flag = 1

    return holo, flag

def gsfocus(H, reginds, precision,zhi=1.2536,zlo=0.230):
    # golden section search focus.
    # inputs:
    # H - hologram or hologram region to focused
    # reginds - if the hologram region has been padded, the indices of the bounding
    #           box of the diffraction pattern of index within H (xmin,ymin,xmax,ymax)
    #           [output from extract_region, in utils.py]
    # precision - desired precision (in m). Typical value 0.001 m
    # zhi - lower limit of imaging volume (physical distance from Camera)
    # zlo - upper limit of imaging volume (physical distance from camera)

    # returns: focused reconstruction and physical focus distance (zf), in m
    
    t = time.time()
    A, params = holofft(H)



    phi = 1/(( 1 + 5**0.5)/2)
    dz = 999
    n = 1
    (xmin,ymin,xmax,ymax) = reginds
    while dz > precision:
        z1 = -(phi*(zhi-zlo) - zhi)
        z2 = phi*(zhi-zlo) + zlo

        im1 = propogate(physical_to_optical(z1), A, params)
        im1 = im1[xmin:xmax,ymin:ymax]

        im2 = propogate(physical_to_optical(z2), A, params)
        im2 = im2[xmin:xmax,ymin:ymax]

        y1 = -stdcorr(im1)
        y2 = -stdcorr(im2)

        n = n + 2

        if y2 > y1:
            zhi = z2
        else:
            zlo = z1

        dz = np.abs(zhi - zlo)

    if y1 < y2:
        zf = z1
        im = im1
    else:
        zf = z2
        im = im2

    elapsed = time.time() - t

    return im, zf

def stdcorr(img):
    # calculate standard correlation focus metric for a given image/reconstruction
    # works for numpy array or pytorch tensor

    M,N = img.shape
    if isinstance(img,np.ndarray):
        ug = np.mean(img)
        img = img/ug
        f = (1/(M*N))*np.sum(img[1:,1:]*img[:-1,:-1] - (1/(M*N*np.square(ug))))
    else:
        ug = torch.mean(img)
        img = img/ug
        f = (1/(M*N))*torch.sum(img[1:,1:]*img[:-1,:-1] - (1/(M*N*torch.square(ug))))
    return f
