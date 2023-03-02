function [F,B,alpha]=solve(mu_F,Sigma_F,mu_B,Sigma_B,C,sigma_C,alpha_init,maxIter,minLike)

% SOLVE     Solves for F,B and alpha that maximize the sum of log
%   likelihoods at the given pixel C.
%   input:
%   mu_F - means of foreground clusters (for RGB, of size 3x#Fclusters)
%   Sigma_F - covariances of foreground clusters (for RGB, of size
%   3x3x#Fclusters)
%   mu_B,Sigma_B - same for background clusters
%   C - observed pixel
%   alpha_init - initial value for alpha
%   maxIter - maximal number of iterations
%   minLike - minimal change in likelihood between consecutive iterations
%
%   returns:
%   F,B,alpha - estimate of foreground, background and alpha
%   channel (for RGB, each of size 3x1)
%
    
% Initialize identity matrix
I = eye(3);

% Initialize empty array to store results
vals = [];

% Loop over all factor means
for i = 1:size(mu_F, 2)
    % Get current factor mean and its inverse covariance matrix
    mu_Fi = mu_F(:, i);
    invSigma_Fi = inv(Sigma_F(:, :, i));
    
    % Loop over all idiosyncratic means
    for j = 1:size(mu_B, 2)
        % Get current idiosyncratic mean and its inverse covariance matrix
        mu_Bi = mu_B(:, j);
        invSigma_Bi = inv(Sigma_B(:, :, j));
        
        % Initialize alpha and iteration counter
        alpha = alpha_init;
        iter = 1;
        
        % Initialize likelihood to negative infinity
        lastLike = -realmax;
        
        % Rest of code goes here...

        while (1)
            
            % solve for F,B
%             A=[invSigma_Fi+I*(alpha^2/sigma_C^2) , I*alpha*(1-alpha)/sigma_C^2; 
%                I*((alpha*(1-alpha))/sigma_C^2)  , invSigmabi+I*(1-alpha)^2/sigma_C^2];
%              
%             b=[invSigma_Fi*mu_Fi+C*(alpha/sigma_C^2); 
%                invSigmabi*mubi+C*((1-alpha)/sigma_C^2)];
            % Construct matrix A and vector b for solving for F and B
            A = [invSigma_Fi + I * (alpha^2 / sigma_C^2),...
                I * alpha * (1 - alpha) / sigma_C^2;...
                I * ((alpha * (1 - alpha)) / sigma_C^2),...
                invSigma_Bi + I * (1 - alpha)^2 / sigma_C^2];
            
            b = [invSigma_Fi * mu_Fi + C * (alpha / sigma_C^2);...
                invSigma_Bi * mu_Bi + C * ((1 - alpha) / sigma_C^2)];

           
            X=A\b;
            % Extract F and B from the solution
            % Ensure that all values are between 0 and 1
            F = max(0, min(1, X(1:3)));
            B = max(0, min(1, X(4:6)));
            
            % Compute alpha using quadratic approximation
            numerator = (C - B)' * (F - B);
            denominator = sum((F - B).^2);
            alpha = max(0, min(1, numerator / denominator));
            
            % Compute negative log-likelihood of the model
            L_C = -sum((C - alpha * F - (1 - alpha) * B).^2) / sigma_C;
            
            % Compute negative log-likelihood of the factor model
            L_F = -((F - mu_Fi)' * invSigma_Fi * (F - mu_Fi)) / 2;
            
            % Compute negative log-likelihood of the idiosyncratic model
            L_B = -((B - mu_Bi)' * invSigma_Bi * (B - mu_Bi)) / 2;
            
            % Compute total log-likelihood
            like = L_C + L_F + L_B;

           
            % Check if maximum number of iterations has been reached or if likelihood hasn't improved
            if iter >= maxIter || abs(like - lastLike) <= minLike
               break;
            end
            
            % Update last likelihood and iteration count
            lastLike = like;
            iter = iter + 1;

        end
        
        % Store current values of F, B, alpha, and likelihood in a struct
        val.F = F;
        val.B = B;
        val.alpha = alpha;
        val.like = like;
        
        % Append current struct to the array of results
        vals = [vals, val];

    end
end

% Find index of maximum likelihood estimate
[t, ind] = max([vals.like]);

% Extract F, B, and alpha corresponding to maximum likelihood estimate
F = vals(ind).F;
B = vals(ind).B;
alpha = vals(ind).alpha;


