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
import time



def link(costs,nl,nr):
    """given a matrix of costs where the rows are representative of trajectories tl and the columns of trajectories tr, output lists of paired indices into tl and tr"""
    from scipy.optimize import linear_sum_assignment
    t0 = time.time()
    # find the links with the munkres algorithm
    inds = linear_sum_assignment(costs)
    # now remove all null links
    inds = np.array(inds).T[(inds[0]<nl)&(inds[1]<nr)]
    # and return the results
    print('%d links generated from %d possibilities in %d seconds.'%(len(inds),min(nl,nr),time.time()-t0))
    return inds.T

triangulation_mats = np.load('./triangulation_and_transformation_matrices.npy').item()
P1 = triangulation_mats['P1']
P2 = triangulation_mats['P2']

camera_to_world = np.load('./camera_to_world.npy').item()
R = camera_to_world['r']
T = camera_to_world['t']

def triangulate(link_inds,tl,tr):
    """given a list of links, triangulate out 3d position and return array of points with format [pointslink0,pointslink1,pointslink2,...]."""
    links = [[tl[i],tr[j]] for i,j in link_inds.T]
    import cv2
    worldpts = []
    for l,r in links:
        il,ir = shared_indices(l,r)
        lpos = np.flip(l.positions[il],axis=1) # input points should be the horizontal coordinate first 
        rpos = np.flip(r.positions[ir],axis=1) 
        cam = cv2.convertPointsFromHomogeneous(cv2.triangulatePoints(P1,P2,lpos.T,rpos.T).T)/10.0 # 3d positions in cm
        cam = cam[:,0,:] # remove that extra axis. These are points from the camera frame
        exp = np.array([np.dot(R,p)+T for p in cam])
        worldpts.append(exp)
    return np.array(worldpts)


def shared_indices(l,r): 
    """this is the set of times shared between two trajectories l and r """
    times1 = l.times
    times2 = r.times
    out = np.array([[i, j]  for i, x in enumerate(times1) for j, y in enumerate(times2) if x == y]).T
    if len(out)==0:
        return [],[]
    else:
        return out

def cost(tl,tr,flag='not_dummy'): 
    """ generate the cost associated with linking tl to tr"""
    
    sig_l2= 10.0
    sig_tv= 1.0 
    l2_max = 25.0
    tv_max = 2.0
    f = 0.7 # 70 percent of tracks must be shared 
    
    if flag=='dummy':
        p = np.exp(-l2_max**2/sig_l2**2)
        r = np.exp(-tv_max**2/sig_tv**2)
        return p*r

    else:
        
        il,ir = shared_indices(tl,tr)
        if len(il)>0:
            
            # first get the positions at each time shared between the trajectories
            yl,xl = tl.positions[il].T
            yr,xr = tr.positions[ir].T

            #(1) now calculate the epipolar deviation cost 
            dy_sq = np.mean(np.square(np.subtract(yl,yr)))
            p = np.exp(-dy_sq/sig_l2**2)

            #(2) now calculate the disparity cost which is binary if disparity is physical
            max_disp = 500.0
            disp = np.sqrt(np.mean(np.square(np.subtract(xl,xr))))
            if disp < max_disp:
                q = 1.0
            else:
                q = 0.0

            #(3) now calculate the TV cost
            dl = np.subtract(yl,np.roll(yl,-1))
            dr = np.subtract(yr,np.roll(yr,-1))
            dtv_sq = np.mean(np.square(np.subtract(dl,dr)))
            r = np.exp(-dtv_sq/sig_tv**2)
            
            # (4) require that at least fraction f of the points in the shorter trajectory are 
            # held in common with points in the longer trajectory  
            nl = len(tl.positions) # number of points in left trajectory 
            nr = len(tr.positions) #number of points in right trajectory 
            n = min(nl,nr) # number of points in the shorter trajectory
            nc = len(il) # number of common points 
            if nc > f*n:
                s = 1.0
            else: 
                s = 0.0
                
            # require that all trajectories span at least 200 pixels 
            if xl.max()-xl.min()>200 and xr.max()-xr.min()>200:
                t = 1.0
            else:
                t= 0.0 
            return p*q*r*s*t
        
        else:
            return 0.0 

def cart_cross(x,y):
    """Cartesian cross product of two vectors"""
    #return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]).reshape(len(x),len(y),2)
    #https://stackoverflow.com/questions/51905176/generating-matrix-of-pairs-from-two-object-vectors-using-numpy/51905228?noredirect=1#comment90761191_51905228
    return np.transpose(np.meshgrid(x, y), (2, 1, 0))
    
    
def cost_matrix(tl,tr,t_max=0.5):
    """given arrays of left and right trajectory objects tl and tr with nl = len(tl) and nr = len(tr),
    generate an (nl + nr) x (nl + nr) matrix where the upper left nlxnr blocks C_ij are the costs of linking trajectory
    tl[i] to trajectory tr[j]. The upper right nl x nl block is a diagonal matrix allowing each trajectory from the left an opportunity
    to null link. The bottom left nr x nr block is a diagonal matrix allowing each trajectory from the right a null link
    opportunity. The bottom right nr x nl block is a grid matrix where every element allows any null link to pair with any other."""

    
    import time
    t0 = time.time()
    from numpy import ma
    nl = len(tl)
    nr = len(tr)
    l_traj_times = np.fromiter(map(lambda t: t.T, tl),dtype=float)
    r_traj_times = np.fromiter(map(lambda t: t.T, tr),dtype=float)
    mask = np.abs(l_traj_times.reshape(-1, 1) - r_traj_times) < t_max # here's the mask saying medians are close enough 
    i1,j1 = ma.MaskedArray.nonzero(ma.array(mask)) # the indices cooresponding to nonzero data in cost matrix
    
    # now make a new sparse matrix with shape of map which is filled with a cost(tl[i],tr[j]) if mask[i,j] is true
    traj_mat = cart_cross(tl,tr) # matrix of pairs of trajectories    
    
    # this is the set of all costs 
    cost_data1 = np.fromiter(map(lambda q: cost(q[0],q[1]), traj_mat[i1,j1]),dtype=float) # this generates the costs 

    # now generate all indices associated with a null link possibility

    # bottom right
    i2,j2 = np.indices((nr,nl))
    i2,j2 = i2.flatten()+nl,j2.flatten()+nr
    cost_data2 = np.ones(i2.size,dtype=float)*cost(None,None,'dummy')
    
    # top right
    i3 = np.fromiter(range(nl),dtype=int)
    j3 = np.fromiter(range(nl),dtype=int)
    j3 = j3 + nr 
    cost_data3 = np.ones(i3.size,dtype=float)*cost(None,None,'dummy')
    
    # bottom left
    i4 = np.fromiter(range(nr),dtype=int)
    j4 = np.fromiter(range(nr),dtype=int)
    i4 = i4 + nl 
    cost_data4 = np.ones(i4.size,dtype=float)*cost(None,None,'dummy')

    # now make the full set of cost data for the matrix generation
    cost_data = np.concatenate((cost_data1,cost_data2,cost_data3,cost_data4))
    i = np.concatenate((i1,i2,i3,i4))
    j = np.concatenate((j1,j2,j3,j4))    
    
    # now you have the cost matrix and all real links are filled with their proper values
    N = nl+nr #dimension of output square matrix 
    costs = scipy.sparse.csc_matrix((cost_data,(i,j)),shape = (nl+nr,nl+nr),dtype=float) # and this puts them in a sparse matrix
    print('%d x %d cost matrix generated in %d seconds'%(nl+nr,nl+nr,time.time()-t0))

    costs = costs.toarray()

    return -costs


class Track:
    def __init__(self,l,r,positions): # link is a pair [l,r] of trajectories 
        il,ir = shared_indices(l,r)
        self.positions_l = l.positions[il]
        self.positions_r = r.positions[ir]
        self.times = l.times[il]
        self.positions = positions
        dt = l.times[1]-l.times[0]
        self.velocities = np.diff(positions,1,axis=0)/dt
        self.accelerations = np.diff(positions,2,axis=0)/dt**2
    def __len__(self):
        return len(self.times)
    def __repr__(self):
        posi = str(tuple(np.around(self.positions[0],2)))
        posf = str(tuple(np.around(self.positions[-1],2)))
        duration = np.around(np.amax(self.times)-np.amin(self.times),2)
        return 'track from ' + posi + 'cm to ' + posf + 'cm in %2f s'%duration
    
    
    
def links_to_tracks(link_inds,tl,tr):
    out = []
    positions_set = triangulate(link_inds,tl,tr)
    for i in range(len(positions_set)):
        l = tl[link_inds[0]][i]
        r = tr[link_inds[1]][i]
        p = positions_set[i]
        out.append(Track(l,r,p))
    return out