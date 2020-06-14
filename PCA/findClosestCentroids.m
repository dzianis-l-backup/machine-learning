function idx = findClosestCentroids(X, centroids)
% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

for i=1:size(X,1)
  idx(i) = getClusterIndex(X(i,:), centroids);
endfor


end

function clusterIndex = getClusterIndex(x, centroids)
  dist = Inf;
  clusterIndex = 1;
  for j=1:size(centroids, 1)
       currDist = getDistance(centroids(j, :), x);
       if currDist < dist
         clusterIndex = j;
         dist = currDist;
       endif
  endfor
endfunction

function dist = getDistance(centr,x)
  dist = sum ( (x-centr).^2 );
endfunction

