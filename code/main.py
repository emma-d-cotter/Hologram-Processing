from utils import *
from detection import *
from params import *
from reconstruction import *
import cv2

# this sample script performs diffraction pattern detection and auto-focusing for a sample hologram
# note - background subtraction and classification for optical turbulence are not performed in this script.

imagepath = r'..\sample_data\sample.jpg' # sample background subtracted hologram
holo = cv2.imread(imagepath,cv2.IMREAD_GRAYSCALE)
holo = holo - 104 # to store background-subtracted hologram as jpg, had to shift to all positive values
mask = create_lpf_mask(holo,keep_frac = 0.5)
regions = lpf_detection(holo, mask) # using default parameters

pad_size = 2.5 # mm
precision = 0.001 # m
outpath = r'..\extracted_regions'
for i, region in enumerate(regions):
  H,reginds,_ = extract_region(holo,region.bbox,pad_size)
  im, zf = gsfocus(H,reginds, precision)
  (xmin,ymin,xmax,ymax) = region.bbox

  outfile = os.path.join(outpath,'sample' + '_' + str(xmin).zfill(4) + '_' + str(ymin).zfill(4)  + '_' + str(xmax).zfill(4) + '_' + str(ymax).zfill(4)  + '_' + str(np.round(zf*1000)).zfill(4) + '.jpg')
  cv2.imwrite(outfile,im.cpu().numpy())
