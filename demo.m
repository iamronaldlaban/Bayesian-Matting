clc
clear 
close all

% To demonstrate Bayesian Matting 

%  Read the input image, trimap and the ground truth 
input=imread('input_training_lowres\GT15.png');
% im = imresize(im, 0.5);
trimap=imread('trimap_training_lowres\Trimap1\GT15.png');
% trimap = imresize(trimap, 0.5);
ground_truth = imread('gt_training_lowres\GT15.png');
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

[F,B,alpha] = get_Bayesmat(input,trimap,P);

figure;
subplot(1, 3, 1), imshow(input);
title('Input');

subplot(1, 3, 2), imshow(trimap);
title('Trimap');

subplot(1, 3, 3), imshow(alpha);
title('Alpha Matte');


% Image compositing 
background = im2double(imread('background.jpg'));
[h, w, c] = size(input);
background = imresize(background, [h, w]);

% Image Compositing Equation
composite = alpha .* im2double(input) + (1 - alpha) .* background;

figure;
subplot(2, 2, 1), imshow(input);
title('Foreground');

subplot(2, 2, 2), imshow(alpha);
title('Alpha Matte');

subplot(2, 2, 3), imshow(background);
title('Background');

subplot(2, 2, 4), imshow(composite);
title('Composite Image using Bayesian Matting');

%  Laplacain Matting 

[L_alpha]=get_Laplacian(input,trimap);

figure;
subplot(1, 3, 1), imshow(input);
title('Input');

subplot(1, 3, 2), imshow(trimap);
title('Trimap');

subplot(1, 3, 3), imshow(L_alpha);
title('Laplacian Alpha Matte');


% Image Compositing Equation
composite = L_alpha .* im2double(input) + (1 - L_alpha) .* background;

figure;
subplot(2, 2, 1), imshow(input);
title('Foreground');

subplot(2, 2, 2), imshow(L_alpha);
title('Laplacian Alpha Matte');

subplot(2, 2, 3), imshow(background);
title('Background');

subplot(2, 2, 4), imshow(composite);
title('Composite Image using Laplacian Matting');

% Comparision of Bayesian Matting and Laplacian Matting with Ground Truth

figure;
subplot(1, 3, 1), imshow(im2double(ground_truth));
title('Ground Truth');

subplot(1, 3, 2), imshow(alpha);
title('Alpha Matte');

subplot(1, 3, 3), imshow(L_alpha);
title('Laplacian Alpha Matte');


% Mean Squared error

[bay_mse]=get_MSE(ground_truth,alpha);

[lap_mse]=get_MSE(ground_truth,L_alpha);


% Sum of Absolute Difference

[bay_sad] = get_SAD(ground_truth, alpha);

[lap_sad] = get_SAD(ground_truth, L_alpha);


% Gradient

[bay_grad] = get_Gradient(ground_truth, alpha);

[lap_grad] = get_Gradient(ground_truth, L_alpha);

