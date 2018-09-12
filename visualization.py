import numpy as np 
import pims
import skimage as ski 
from skimage import morphology, io, color, util, exposure, draw, filters
import os 
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation
from scipy import ndimage
import trackpy as tp
import pandas as pd 
import scipy 
from scipy import interpolate


def wordmask(text,position,size):
    """generate a boolean mask of text.
    text is a string of what you want to type
    position is a tuple (x,y)
    size = 60 is the size of the text"""
    from PIL import Image, ImageDraw, ImageFont
    shape = (720,1680)
    txt = Image.new('L', shape[::-1], (0))
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', size)
    d = ImageDraw.Draw(txt)
    d.text(position, text, font=fnt, fill=True)
    return  np.array(txt,dtype=bool)


def annotate_linked_features(V,feats,i):
    """ using the linked features dataframe feats, return an annotated image
    from float 0-1 single channel pims video V. 
    * feats is a dataframe containing 'x','y','particle','frame'."""
    import skimage as ski
    from skimage import draw
    import matplotlib.cm as cm
    # define colors, get frame and features 
    colors = (cm.cool(np.linspace(0.0, 1.0, 10))[:,:3]*255).astype('uint8')
    ifeats = feats[feats['frame']==i]
    im = V[i]
    im = (255*np.stack((im,)*3,-1)).astype('uint8')
    # draw on the frame 
    words = wordmask('time {} s'.format(round(i/190.0,3)),(20,20),40) #generate mask of the frame number 
    im[words] = [0,0,240]
    
    # draw all features on the frame with circles.. 
    for _,vals in ifeats.iterrows(): 
        y = int(round(vals['x']))
        x = int(round(vals['y']))
        color = colors[int(vals['particle']%10)]
        p = vals['particle']
        row,col = ski.draw.circle_perimeter(x,y,9,shape=(720, 1680))
        im[row,col]=color
        row,col = ski.draw.circle_perimeter(x,y,10,shape=(720, 1680))
        im[row,col]=color
        row,col = ski.draw.circle_perimeter(x,y,11,shape=(720, 1680))
        im[row,col]=color    
        #also add the feature number of each object
        words = wordmask(str(int(p)),(y+5,x+5),size=30)
        im[words]=color
    return im


def draw_trajectory(im,traj,color=[250,0,250]):
    """draw the trajectory object traj onto the image im"""
    import skimage
    from skimage import draw
    if len(im.shape)<3:
        im = (np.stack((im,)*3,-1)*255).astype('uint8')
    for x,y in traj.positions:
        x = int(round(x))
        y = int(round(y))
        r,c = ski.draw.circle(x,y,7,im.shape)
        im[r,c]=color
    return im


def draw_many_trajectories(im,trajs,indices):
    import matplotlib.cm as cm
    # define colors, get frame and features 
    colors = (cm.cool(np.linspace(0.0, 1.0, 20))[:,:3]*255).astype('uint8')
    img = np.array(im)
    if len(img.shape)!=3:
        img = (np.stack((img,)*3,-1)*255).astype('uint8')
    j=0
    for i,c in zip(indices,colors):
        j+=1
        traj = trajs[i]
        img = draw_trajectory(img,traj,color = colors[j%20])
    return img

def draw_track(imL,imR,track,color=[250,0,250],together=True):
    if len(imL.shape)<3:
        imL = (np.stack((imL,)*3,-1)*255).astype('uint8')
    if len(imR.shape)<3:
        imR = (np.stack((imR,)*3,-1)*255).astype('uint8')    
    border = np.zeros((720,20,3),dtype='uint8')
    for x,y in track.positions_l:
        x = int(round(x))
        y = int(round(y))
        r,c = ski.draw.circle(x,y,7,imL.shape)
        imL[r,c]=color
    for x,y in track.positions_r:
        x = int(round(x))
        y = int(round(y))
        r,c = ski.draw.circle(x,y,7,imR.shape)
        imR[r,c]=color        
    if together:
        return np.hstack((imL,border,imR))
    else:
        return imL,imR
    
def draw_many_tracks(imL,imR,tracks,indices):
    if imL is None:
        imL = (np.ones((720,1680,3))*255).astype('uint8')
        imR = imL
    import matplotlib.cm as cm
    # define colors, get frame and features 
    colors = (cm.tab20(np.linspace(0.0, 1.0, len(indices)))[:,:3]*255).astype('uint8')
    border = np.zeros((720,20,3),dtype='uint8')
    j=0
    for i,c in zip(indices,colors):
        j+=1
        track = tracks[i]
        imL,imR = draw_track(imL,imR,track,color=colors[j%20],together=False)
    return np.hstack((imL,border,imR))    

def to_traces(tracks):
    positions = np.array([a for b in tracks for a in b.positions])
    times = np.around(np.array([a for b in tracks for a in b.times]),3)
    utimes = np.sort(np.unique(times))
    traces = []
    for t in utimes: 
        traces.append(positions[times==t])
    return traces

def track_vis_3D(track):
    X = track.positions
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*X.T,color='blue')
    ax.set_xlim3d(-10,50)
    ax.set_ylim3d(0,10)
    ax.set_zlim3d(-5,45)
    ax.set_xlabel('DOWNSTREAM [cm]')
    ax.set_ylabel('VERTICAL [cm]')
    ax.set_zlabel('CROSS STREAM [cm]')
    ax.view_init(-50, -90)
    fig.show()