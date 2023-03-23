% Load the input image and trimap
img = imread('input_training_lowres\GT04.png');
trimap = imread('trimap_training_lowres\Trimap1\GT04.png');

% Convert the image and trimap to double precision and normalize
img = double(img) ./ 255;
trimap = double(trimap) ./ 255;

% Get the size of the image
[m, n, c] = size(img);

% Calculate the foreground, background, and unknown pixels
fg = trimap > 0.99;
bg = trimap < 0.01;
unk = ~(fg | bg);

% Create the Laplacian matrix
A = sparse([], [], [], m*n, m*n, 5*m*n); % initialize sparse matrix
for i = 1:m*n
A(i, i) = 4; % set diagonal element
if i > n % add left neighbor
A(i, i-n) = -1;
end
if mod(i, n) ~= 0 % add right neighbor
A(i, i+1) = -1;
end
if i <= (m-1)*n % add top neighbor
A(i, i+n) = -1;
end
if mod(i, n) ~= 1 % add bottom neighbor
A(i, i-1) = -1;
end
end

% Calculate the alpha matte
alpha = zeros(m, n);
for i = 1:c
b = zeros(m*n, 1); % initialize b vector
b(fg(:, :, i)) = 1; % set foreground pixels to 1
b(bg(:, :, i)) = 0; % set background pixels to 0
alpha(unk(:, :, i)) = A(unk(:, :, i), unk(:, :, i)) \ b(unk(:, :, i)); % solve linear system for unknown pixels
end

% Save the output alpha matte
imshow(alpha);