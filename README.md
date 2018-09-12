# track-em

These codes are for the analysis of a set of experiments measuring te transport of glass beads in an artificial flume using stereoscopic computer vision from a binocular pair of cameras. 

1) Identify moving particles from the background with background_subtraction
2) Link moving particles into 2d trajectories with linking_2d, powered by trackpy 
3) Link pairs of trajectories from left and right views as an assignment problem with linking_3d
	powered by a cython implementation of the Kuhn-Munkres algorithm. Further, triangulate these 		paired 2d trajectories into 3d trajectories using the stereo calibration parameters (obtained 
	from the bouguet matlab toolbox and a lot of struggle) using opencv. 

points for improvement include:
1) a more powerful feature identification based upon the Xue 2017 cell detection paper 
	concerning compressed sensing convolutional neural networks 
2) better interpolation scheme 

some notes: 

camera_to_world.npy are the rotation and translation matrices to invoke the camera to world coordinate
system transform

rectification_parameters.mat are the rotation and tranlation matrices required by the rectifyvids.m script

fast_cost_mat.py includes cython and vectorized numpy operations which appear to replace the analogous linking_3d.py functions much faster 
