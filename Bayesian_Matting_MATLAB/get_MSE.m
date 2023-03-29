function [mse]=get_MSE(ground_truth,alpha)
img1 = im2double(ground_truth(:,:,1));
img2 = alpha;

% Calculate the MSE
if var(img1(:)) == 0 || var(img2(:)) == 0
    mse = 0;
else
    mse = mean(mean((img1 - img2).^2));
end

% mse = sum(sum((img1 - img2).^2), "omitnan") / (numel(img2))