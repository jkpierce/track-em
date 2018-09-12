
addpath	../TOOLBOX_calib/
load rectification_parameters.mat

lpath = './left-test.avi';
rpath = './right-test.avi';

% load left video and generate a container for writing the rectified one out
L = VideoReader(lpath);
LR = VideoWriter('./left-test-rect.avi');
LR.FrameRate = 188.0367;

% same for right video 
R = VideoReader(rpath);
RR = VideoWriter('./right-test-rect.avi');
RR.FrameRate = 190.3952;
%https://stackoverflow.com/questions/19471751/loop-through-video-file-frame-by-frame-in-matlab

% loop through left video and rectify it 
open(LR)
for i=1:10%L.NumberOfFrames
    fprintf('saving frame %d of left vid\n',i); 
    a = l(:,:);
    l = read(L,i);
    l = l(:,:,1);
    l = double(l);
    l = rect(l,R_L,fc_left,cc_left,kc_left,alpha_c_left,KK_left_new);
    l = uint8(l);
    writeVideo(LR,l);
end
close(LR)

open(RR)
for i=1:10 %R.NumberOfFrames
    fprintf('saving frame %d of right vid\n',i);    
    r = read(R,i);
    r = r(:,:,1);
    r = double(r); 
    r = rect(r,R_R,fc_right,cc_right,kc_right,alpha_c_right,KK_right_new);
    r = uint8(r);
    writeVideo(RR,r);
end
close(RR)

%

