function [mu,Sigma]=cluster_OrachardBouman(S,w,minVar)

% Implements color clustering presented by Orchard and Bouman (1991)
%   input:
%   S - measurements vector
%   w - corresponding weights
%   returns:
%   mu - cluster means
%   Sigma - cluster covariances
%
%   Author: Michael Rubinstein
%

% initially, all measurements are one cluster
C1.X=S;
C1.w=w;
C1=calc(C1);
nodes=[C1];

while (max([nodes.lambda])>minVar)
    nodes=split(nodes);
end

for i=1:length(nodes)
    mu(:,i)=nodes(i).q;
    Sigma(:,:,i)=nodes(i).R;
end

% calculates cluster statistics
function C=calc(C)
% calculate weighted mean
C.q = sum(bsxfun(@times, C.X, C.w), 1) / sum(C.w);

% calculate weighted covariance
t = bsxfun(@times, C.X - C.q, sqrt(C.w));
C.R = (t' * t) / sum(C.w) + 1e-5 * eye(3);

% calculate weighted total squared error
C.wtse = sum(sum(bsxfun(@minus, C.X, C.q).^2 .* repmat(C.w, [1, size(C.X, 2)])));

[V,D]=eig(C.R);
C.e=V(:,3);
C.lambda=D(9);

% splits maximal eigenvalue node in direction of maximal variance
function nodes=split(nodes)

% find node with maximum lambda value
[~, max_idx] = max([nodes.lambda]);
max_node = nodes(max_idx);

% split the node into two children nodes
idx = max_node.X * max_node.e <= max_node.q * max_node.e;
child_a.X = max_node.X(idx, :);
child_a.w = max_node.w(idx);
child_b.X = max_node.X(~idx, :);
child_b.w = max_node.w(~idx);

% calculate the statistics of the children nodes
child_a = calc(child_a);
child_b = calc(child_b);

% remove the max node and replace it with its children nodes
nodes(max_idx) = [];
nodes = [nodes, child_a, child_b];


