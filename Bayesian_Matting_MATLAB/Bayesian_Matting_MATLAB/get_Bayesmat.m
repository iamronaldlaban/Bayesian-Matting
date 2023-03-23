function [F,B,alpha]=get_Bayesmat(input,trimap,P)

% The function get_Bayesmat takes an input image (im) and a trimap mask 
% (trimap) with the same dimensions as im. The trimap has three possible 
% values: 0 forbackground pixels, 1 (or 255) for foreground pixels, and 
% all other pixels are considered unknown. The function also takes a 
% parameter structure (P). 
%
% Arguments
% input  : Input image M x N array.
% trimap : Trimap image M x N array.
% P : Parameter structure as defined in demo.m
% Return values
% F : Foreground M x N array
% B : Background M x N array
% alpha : Alpha matte M x N array

% Convert image parameters from uint8 to double
im = double(input);
im = im / 255;
trimap = double(trimap); 
trimap = trimap / 255;


bg_mask = trimap ==0; % background region mask
fg_mask = trimap == 1; % foreground region mask
unk_mask = ~bg_mask & ~fg_mask; % unknow region mask

% initialize F,B,alpha
F = im; F(repmat(~fg_mask, [1, 1, 3])) = 0;
B = im; B(repmat(~bg_mask, [1, 1, 3])) = 0;
alpha = zeros(size(trimap));
alpha(fg_mask) = 1;
alpha(unk_mask) = NaN;

nUnknown_pts = sum(unk_mask(:));


% square structuring element for eroding the unknown region(s)
% g=fspecial('gaussian', P.N_box, P.sigma); g = g / max(g(:));
se=strel('square', 20);

n = 1;
unk_reg = unk_mask;

max_li = 10;
mi =0;
while n<nUnknown_pts
    % guassian falloff. will be used for weighting each pixel neighborhood
    g=fspecial('gaussian', P.N_box, P.sigma); g = g / max(g(:));
    % get unknown pixels to process at this iteration
    unk_reg=imerode(unk_reg,se);
    unk_pixels = ~unk_reg & unk_mask;
    [Y, X] = find(unk_pixels); 
    rep_max = 0;
     for i = 1:length(Y)
         
        % take current pixel
        x = X(i); y = Y(i);
        cur_pixel = reshape(im(y,x,:), [3,1]);

        % take surrounding alpha values
        a = get_box(alpha, x, y, P.N_box);
        
        % take surrounding foreground pixels
        fore_pixels = get_box(F, x, y, P.N_box);
        fore_weights = (a.^2) .* g;
        fore_pixels = reshape(fore_pixels, P.N_box * P.N_box, 3);
        fore_pixels = fore_pixels(fore_weights > 0, :);
        fore_weights = fore_weights(fore_weights > 0);
        
        % take surrounding background pixels
        bck_pixels = get_box(B, x, y, P.N_box);
        bck_weights = ((1 - a).^2) .* g;
        bck_pixels = reshape(bck_pixels, P.N_box * P.N_box, 3);
        bck_pixels = bck_pixels(bck_weights > 0, :);
        bck_weights = bck_weights(bck_weights > 0);
        
        % if not enough data, return to it later...
        if length(fore_weights)<P.minN || length(bck_weights)<P.minN
            rep_max = rep_max + 1 ;
            if (rep_max == length(Y))
                mi = mi + 1;
                if (mi == max_li)
                    mi = 0;
                    P.N_box = P.N_box + 10;
                    if (P.N_box == length(im) / 4 )
                        n = nUnknown_pts;
                    end
                end
            end
         continue;
        end
        
        % partition foreground and background pixels to clusters (in a
        % weighted manner)
        [mu_fore, Sigma_fore] = cluster_OrachardBouman(fore_pixels, fore_weights, P.clust.minVar);
        [mu_bck, Sigma_bck] = cluster_OrachardBouman(bck_pixels, bck_weights, P.clust.minVar);
        
        % update covariances with camera variance, as mentioned in their
        % addendum
        Sigma_fore = addCamVar(Sigma_fore, P.sigma_C);
        Sigma_bck = addCamVar(Sigma_bck, P.sigma_C);
        
        % set initial alpha value to mean of surrounding pixels
        alpha_init = nanmean(a(:));
        
        % solve for current pixel
        [f,b,a]=solve(mu_fore,Sigma_fore,mu_bck,Sigma_bck,cur_pixel,P.sigma_C,alpha_init,P.opt.maxIter,P.opt.minLike);
        
        F(y,x,:)=f;
        B(y,x,:)=b;
        alpha(y,x)=a;
        unk_mask(y,x)=0; % remove from unkowns
        n=n+1;
    end
 
end
% retruns the surrounding N-rectangular neighborhood of matrix m, centered
% at pixel (x,y) 
function r=get_box(m,x,y,N)

[h,w,c]=size(m);
halfN = floor(N/2);
n1=halfN; n2=N-halfN-1;
r=nan(N,N,c);
xmin=max(1,x-n1);
xmax=min(w,x+n2);
ymin=max(1,y-n1);
ymax=min(h,y+n2);
pxmin=halfN-(x-xmin)+1; pxmax=halfN+(xmax-x)+1;
pymin=halfN-(y-ymin)+1; pymax=halfN+(ymax-y)+1;
r(pymin:pymax,pxmin:pxmax,:)=m(ymin:ymax,xmin:xmax,:);


% finds the orientation of the covariance matrices, and adds the camera
% variance to each axis
function Sigma=addCamVar(Sigma,sigma_C)

Sigma=zeros(size(Sigma));
for i=1:size(Sigma,3)
    Sigma_i=Sigma(:,:,i);
    [U,S,V]=svd(Sigma_i);
    Sp=S+diag([sigma_C^2,sigma_C^2,sigma_C^2]);
    Sigma(:,:,i)=U*Sp*V';
end



