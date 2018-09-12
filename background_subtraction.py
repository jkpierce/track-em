import pims
import numpy as np 
import PIL
import pandas as pd
import skimage
from skimage import measure
import time 
from scipy import ndimage 

# load videos for analysis 
@pims.pipeline
def arr(vid):
    """convert a pims sequence of uint8 255 1-channel images into float 0-1 numpy arrays"""
    return 1/255.0*np.array(vid).astype('float')

# crop videos 
@pims.pipeline
def crop(vid,i0=245,i1=580):
    """vertically crop the image from i0 to i1"""
    return vid[i0:i1,:]

# generate foregrounds
def gen_foregrounds(V,path,name,lr=0.01,thresh=0.2):
    """
    Produce a folder full of foreground images by median background subtraction.
    * V is the input video to be analyzed which is a dtype float 1 channel video on 0-1
    * Folder is at location path and images are named 'name-*.tiff'
    * lr is the learning rate of the background maintenance
    * thresh is the percent difference between background and foreground
    """
    B = np.median(Lc[:200],axis=0) # find initial background 
    for t,V in enumerate(Lc):
        B = (1-lr)*B + lr*V
        V -= B
        mask = np.abs(V)>thresh
        V[mask]=255.0
        V[~mask]=0.0
        V = V.astype('uint8')
        V = PIL.Image.fromarray(V)
        V.save(path+'/'+name+'-%06d.tiff'%t) 
        if t%1000==0:
            print('image {} saved'.format(t))
            
def locate_features(V, filename):
    """ run connected region detection on the video V. 
    Save all detected features to filename
    """
    t0 = time.time()
    features = pd.DataFrame()
    for num, img in enumerate(V):
        if num%1000==0:
            print('frame {} in {} seconds'.format(num,round(time.time()-t0,2)))
        img = ndimage.binary_dilation(img)
        label_image = skimage.measure.label(img, background=0)
        for region in skimage.measure.regionprops(label_image, intensity_image=img):
            if region.area > 60:
                features = features.append([{'y': region.centroid[0],
                                             'x': region.centroid[1],
                                             'frame': num,
                                             },])
    features.to_hdf(filename, key='df', mode='w')
