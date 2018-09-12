import numpy as np 
import pims
import skimage as ski 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import pandas as pd
import trackpy as tp
from trackpy import predict
import pickle 
from scipy import interpolate


class Trajectory:
    """This class represents a \ of a particle. 
    it either intakes a pandas df with frames,x,y keys from the generate_trajectories method,
    or it intakes a dict of positions, frames, and times as might come from 
    the output of trajectory_instance.__dict__ (from the Trajectory class)"""

    def __init__(self, particledata, side):
        pdats = particledata 

        if side=='L':
            fps = 188.04
        elif side=='R':
            fps = 190.40
        else:
            raise ValueError('side is either L or R')

        if isinstance(pdats, pd.DataFrame):
            self.positions = np.array([pdats['y'].values,pdats['x'].values]).T
            self.frames = np.array(pdats['frame'])
            self.times =  np.array(pdats['frame'])/fps
        else: 
            self.positions = pdats['positions']
            self.frames = pdats['frames']
            self.times = pdats['times']

        self.side = side
        self.link = None
        
    @property
    def T(self):
        return np.median(self.times)
    
    def __len__(self): 
        return len(self.frames)
    
    def __eq__(self, other):
        return np.alltrue(self.positions==other.positions) and np.alltrue(self.times==other.times)
    
    def __repr__(self):
        length = np.amax(self.positions[:,1])-np.amin(self.positions[:,1])
        return self.side + ' trajectory of length %d pixel '%length + 'at median time %03f s '%self.T

# load all of the trajectories larger than some size into trajectory objects 
def generate_trajectories(frames,side,minlength=7):
    """given a pandas df of linked features, generate a list of trajectories.
    The pandas df should contain keys 'x','y','particle','frame'
    side = 'L' or side = 'R' sets the frame rate appropriate to each.
    minlength is the minimum allowed number of observations of a particle
    for the trajectory to be included. If 7 for example, it was seen
    in 7 frames"""
    trajs = []
    for p in set(frames.particle): # particle index
        pdats = frames[frames['particle']==p] # particledata. This is the input of the trajectory class init 
        ptraj = Trajectory(pdats,side)
        if len(ptraj)>=minlength:
            trajs.append(ptraj)
    return np.array(sorted(trajs,key=lambda t: t.T))

def distance_filter(trajs,dr=100):
    """ filter trajectories that do not have a long enough travel distance dr"""
    def filter_fn(traj,dr=dr):
        """return True if the spatial length of a trajectory between start and end points exceeds dr"""
        positions = traj.positions
        mins = np.amin(positions,axis=0)
        maxs = np.amax(positions,axis=0)
        return np.linalg.norm(mins-maxs)>dr
    return np.array([t for t in trajs if filter_fn(t)])

def interp_trajectories(trajs,side):
    """ a trajectory is a sample of the continuous particle motion at a given rate.
    The left and right frame rates do not match. This function interpolates the samples
    into continous splines, then resamples the splines at a common framerate of 1000.0fps
    It is pretty slow. It will return a set of trajectories resampled at 1000.0fps. 
    
    Inputs include trajs a numpy array full of Trajectory instances, and side = 'L' or 'R' 
    which fixes the frame rate."""
    if side=='L':
        fps = 188.04
    elif side=='R':
        fps = 190.40
    else:
        raise ValueError('side is either L or R')
    
    out = []
    for ind,t in enumerate(trajs):
        if ind%100==0:
            print('trajectory {} processed'.format(ind), end='\r')
        # get particle data from t 
        positions = t.__dict__['positions']
        times = t.__dict__['times']
        frames = t.__dict__['frames']
        # now interpolate these positions through time 
        xterp = interpolate.interp1d(times,positions[:,0],kind='cubic')
        yterp = interpolate.interp1d(times,positions[:,1],kind='cubic')
        pos_terp = lambda t: np.array([xterp(t),yterp(t)])
        
        # now get the range of t across the trajectory 
        # this is interpolating at 1000 fps from above t0 to below t1
        fps_common = 1000.0
        t0 = min(times)
        t1 = max(times)
        t02 = np.floor(t0)
        t12 = np.ceil(t1)
        otimes = np.linspace(t02,t12,(t12-t02)*fps_common)

        o_times = otimes[np.bitwise_and(otimes>t0,otimes<t1)]
        o_positions = np.array([pos_terp(t) for t in o_times])
        o_frames = None
        
        pdats = dict()
        keys = ['times','positions','frames']
        vals = [o_times,o_positions,o_frames]
        for k,v in zip(keys,vals):
            pdats[k]=v
        
        o = Trajectory(pdats,side)
            
        out.append(o)
    return np.array(out)