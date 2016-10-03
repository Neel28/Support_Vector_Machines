function [dist, d] = dist_metric(X, Y, M)
% a function function that calculate the generic distance between X and Y
% datasets. 
% The distance is defined by metric M. 

% input:
% X: d*n, d-dimensional dataset of n samples
% Y: d*m, d-dimensional dataset of m samples
% M: d*d, distance matric (dist(x,y) = (x-y)^T*M*(x-y))

% output: 
% dist: n*m, pairwise distance
% d: median of dist ( diagonal of dist matrix excluded)

n = size(X,2);
m = size(Y,2);

dist = repmat( diag(X'*M*X), 1, m) - 2 * X'*M*Y + repmat( diag(Y'*M*Y)', n, 1 );
idx = 1:(n+1):n*m;
dist2 = dist(:);
dist2(idx) = [];
d = median(dist2);

