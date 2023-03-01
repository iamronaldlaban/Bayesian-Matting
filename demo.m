clc
clear 
close all

% To demonstrate Bayesian Matting 

%  Read the input image, trimap and the ground truth 
input=imread('input_training_lowres\GT21.png');
% im = imresize(im, 0.5);
trimap=imread('trimap_training_lowres\Trimap1\GT21.png');
% trimap = imresize(trimap, 0.5);
ground_truth = imread('gt_training_lowres\GT21.png');
% ground_truth = imresize(ground_truth, 0.5);

%  Define Parameter
P=struct();

P.N_box   =   100;     % pixel box neighborhood size
P.sigma   =   8;      % variance of gaussian for spatial weighting
P.sigma_C =   0.01;   % camera variance
P.minN    =   10;     % minimum required foreground and background neighbors for optimization
P.guiMode =   0;      % if 1, will show a nice looking progress bar. if 0, will print progress to command line

% clustering parameters
P.clust.minVar    = 0.05;           % minimal cluster variance in order to stop splitting

% optimization parameters
P.opt.maxIter =  50;                % maximal number of iterations
P.opt.minLike =  1e-6;              % minimal change in likelihood between consecutive iterations

%  To Determine Foreground(F), Background(G) and alpha matte (alpha)

[F,B,alpha] = get_Bayesmat(input,trimap,P)

figure;
imshow(input);
title('Input');

figure;
imshow(trimap);
title('Trimap');

figure;
imshow(alpha);
title('Alpha');


% Compositing 

