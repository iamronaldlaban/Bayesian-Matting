function [grad_diff] = get_Gradient(ground_truth, alpha)
img1 = im2double(ground_truth(:,:,1));
img2 = alpha;

% Compute the gradient in x and y directions for each image
[Gx1, Gy1] = imgradientxy(img1);
[Gx2, Gy2] = imgradientxy(img2);

% Compute the gradient magnitude for each image
mag1 = sqrt(Gx1.^2 + Gy1.^2);
mag2 = sqrt(Gx2.^2 + Gy2.^2);

% Compute the sum of the gradient magnitude difference
grad_diff = sum(abs(mag1 - mag2), 'all');

[Gmag,Gdir] = imgradient(img1,img2);

figure, imshowpair(Gmag,Gdir,'montage')
title('Gradient Magnitude (Left) and Gradient Direction (Right)')