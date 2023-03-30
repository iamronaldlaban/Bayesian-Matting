function [sad] = get_SAD(ground_truth, alpha)
img1 = im2double(ground_truth(:,:,1));
img2 = alpha;

% Compute the SAD
sad = sum(abs(img1 - img2), 'all');

% sad = sum(sum(imabsdiff(img1, img2)))