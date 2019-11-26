function [J, grad] = costFunction(theta, X, y)

    m = size(X,1);

    J = -1/m * sum(y .* log(siglog(X, theta)) + (1.-y) .* log(1 .- siglog(X, theta)))

    grad = 1/m * sum( X .* ( siglog(X, theta) .- y ) )'

end

function g = siglog(X, theta)

for i = 1:size(X,1)
    thetaX = theta'*X(i,:)';
    g(i) = 1 / (1 + exp(-thetaX));
end

g = g';

end

