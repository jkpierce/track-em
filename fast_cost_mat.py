import numpy as np 
import time 

def gen_T_mask(tL,tR,nl,nr,t_max = 0.3):
    """generate mask saying whether medians of tL and tR are close enough to potentially pair""" 
    l_traj_times = np.fromiter(map(lambda t: t.T, tL),dtype=float)
    r_traj_times = np.fromiter(map(lambda t: t.T, tR),dtype=float)
    mask = np.abs(l_traj_times.reshape(-1, 1) - r_traj_times) < t_max # here's the mask saying medians are close enough 
    return mask 

def shared_ind_range(l,r,minlength=20,f = 0.7):
    """ generate a tuple ((i0),j0),(r). fix later
    minlength is the number of observations of the particle shared between trajectories.
    f is the percent of tracks from the shorter of the two trajectories which must be shared."""
    x = l.times
    y = r.times #https://stackoverflow.com/questions/52285058/fastest-method-to-get-indices-of-first-and-last-common-elements-between-lists/52285320#52285320
    c = set(x).intersection(y)  # O(n) time
    if c and len(c)>minlength:
        def get_first(l):
            return next(idx for idx, elm in enumerate(l) if elm in c)  # O(n) time
        lslc = (get_first(x), len(x) - get_first(x[::-1]) - 1)
        rslc = (get_first(y), len(y) - get_first(y[::-1]) - 1)
        ntl = len(x)
        ntr = len(y)
        n = min(ntl,ntr)
        if float(lslc[1]-lslc[0])/n > f and float(rslc[1]-rslc[0])/n > f:
            return (lslc,rslc)
        else:
            return None
    else: 
        return None 
    
    
def gen_dist_mats(inds,mask,tL,tR,shared_inds,nl,nr,max_disp=500,min_dist = 200):
    """
    * inds is np.array(np.meshgrid(np.arange(nl),np.arange(nr))).T where nl = len(tL) and nr = len(tR)
    * mask is a boolean matrix describing which elements (i,j) of inds describe pairings tL[i] to tR[j]
      which have median times close enough to pair 
    * tL and tR are lists of trajectory objects for left and right views of the experiment.
    * shared inds [shared_ind_range(*v) for v in traj_mat[mask_T]]
    """
    I,J = inds[mask].T
    epidevs = [] # this will be filled with all values of mean squared vertical deviation between trajectories for everyhwere mask is true
    disps = [] # this will be filled with all values of mean squared disparity between trajectories for everywhere mask is true 
    tvs = [] # this will be filled with all values of mean squared Total Variation norm between trajectories for everyw h fuck
    mindists = [] # these are vals True / False as to whether trajectory is at least min_dist pixels long 
    for si, i,j in zip(shared_inds,I,J):
        if si: # if there are shared elements of the left and right trajectory time series
            ((i0l,iNl),(i0r,iNr)) = si # these are the start and end indices into the timeseries of left and right trajectories which are shared 
            lpos_y, lpos_x = tL[i].positions[i0l:iNl].T  # extract coordinates of the left trajectory along the shared moments in the time series
            rpos_y, rpos_x = tR[j].positions[i0r:iNr].T # and for the right trajectory 
            ######################
            # calc meansq disparity
            disp = np.sqrt(((lpos_x-rpos_x)**2).mean()) # this is mean squared deviation in x direction along the time trace 
            #######################
            # calc meansq epipolar deviation
            y_sq = ((lpos_y-rpos_y)**2).mean() # this is mean squared deviation in y direction (vertical in the image) along the time trace 
            ########################
            # calc meansq total variation 
            dl = np.subtract(lpos_y,np.roll(lpos_y,-1)) # and this is the TV norm with one outlier point wihch probably averages out.. 
            dr = np.subtract(rpos_y,np.roll(rpos_y,-1)) # might want to improve this later and ignore the end of the roll. 
            tv = ((dl-dr)**2).mean() # mean sq 
            ###########################
            # calc boolean vals as to whether particle spans at least min_dist pixels in x dir
            mindists.append(lpos_x.max()-lpos_x.min()>min_dist and rpos_x.max()-rpos_x.min()>min_dist) # if both trajectories span min_dist pixels 
            # append these into the value lists for the matrix development 
            epidevs.append(y_sq) 
            disps.append(disp < max_disp)
            tvs.append(tv)
        else:  # if there are not enough shared pionts between left and right trajectories 
            disps.append(False) # then all of these vals are False and infinity or whatever 
            epidevs.append(np.inf)
            tvs.append(np.inf)
            mindists.append(False)
    # now reshape all of these into matrices 
    Dmat = np.zeros((nl,nr),dtype=bool) # disparity mat   
    Dmat[mask] = disps 
    Dmat[~mask] = False 
    
    DDmat = np.zeros((nl,nr),dtype=bool)
    DDmat[mask] = mindists# mindist mat
    DDmat[~mask]=False
    
    L2mat = np.zeros((nl,nr)) #epipolar deviation mat 
    L2mat[mask] = epidevs   
    L2mat[~mask] = np.inf
    
    tvmat = np.zeros((nl,nr))
    tvmat[mask]=tvs
    tvmat[~mask]=np.inf
    return Dmat,L2mat,tvmat,DDmat



def gen_cost_mat(tL,tR):
    """ generate an nl+nr x nl+nr cost matrix of associations between left trajectories in tL and 
    right trajectories in tR. """ 
    import time
    t0 = time.time()
    nl = len(tL)
    nr = len(tR)
    inds = np.array(np.meshgrid(np.arange(nl),np.arange(nr))).T # an array filled with all pairs of indices 
    mask = gen_T_mask(tL,tR,nl,nr) # mask where median time of left and right trajectories are close enough to potentially pair 
    traj_mat = np.array(np.meshgrid(tL,tR)).T # cartesian product object array of left and right trajectories 
    shared_inds = [shared_ind_range(*v) for v in traj_mat[mask]]  # these are the shared indices for each trajectory pair    
    Dmat,mat_L2_sq,mat_TV_sq,DDmat = gen_dist_mats(inds,mask,tL,tR,shared_inds,nl,nr) # compute all of the masks and norm matrices 
    # now update the mask to include L2 norm greater than limit 
    l2_max = 25.0
    tv_max = 2.0 
    mask = mask&(mat_L2_sq<l2_max**2)&Dmat&(mat_TV_sq<tv_max**2)&DDmat  # make a grand mask over all conditions
    # these conditions are 
    # 1. vertical deviation between trajectories is less than the threshold
    # 2. horizontal deviation between trajectories is physical 
    # 3. trajectoreis span at least min_dist pixels  from start to finish 
    # 4. tv between trajectories is below threshold 
    
    # now calculate the costs vectorized
    sig_l2= 10.0
    sig_tv = 1.0
    costs = np.zeros((nl+nr,nl+nr)) # here's the empty cost matrix 
    # top left 
    costs[:nl,:nr][mask] = np.exp(-mat_L2_sq[mask]/sig_l2**2)*np.exp(-mat_TV_sq[mask]/sig_tv**2) # these are the real trajectory to trajectory linking costs 
    dummycost = np.exp(-l2_max**2/sig_l2**2)*np.exp(-tv_max**2/sig_tv**2) # this is the minimum profit for a linking to be favorable over a dummy link 
    # bottom right
    costs[nl:,nr:] = dummycost # these allow dummy to dummy links (pairs of dummies go unlinked) 
    # bottom left 
    costs[nl:,:nr] = np.eye(nr)*dummycost # these allow left to dummy linking (left trajectories go unlinked) 
    # top right 
    costs[:nl,nr:] = np.eye(nl)*dummycost # these allow right to dummy linking (right trajectories go unlinked) 
    # and flip it to costs
    costs = -costs    # it was actually a profit matrix before. 
    print('%d x %d cost matrix generated in %d seconds'%(nl+nr,nl+nr,time.time()-t0))    
    return costs


def cython_link(costs,nl,nr):
    """use the cython munkres algorithm to link. it's faster than scipy linear_sum_assignment by a lot.""" 
    from munkres import munkres
    # this is the cython munkres program from here 
    # https://github.com/jfrelinger/cython-munkres-wrapper
    # on a 1000x1000 matrix it is a 35% speedup 
    t0 = time.time()
    mat = munkres(costs) # do the munkres algorithm in cython
    inds = np.argwhere(mat) # find all indices of True links
    inds = inds[(inds.T[0]<nl)&(inds.T[1]<nr)] # remove dummy links
    print('%d links generated from %d possibilities in %d seconds with CYTHON.'%(len(inds),min(nl,nr),time.time()-t0))
    return inds.T