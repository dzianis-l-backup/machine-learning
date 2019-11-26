function [J, grad] = costFunctionReg(theta, X, y, lambda)

    m = length(y); % number of training examples

    m = size(X,1);

J = -1/m * sum(y .* log(siglog(X, theta)) + (1.-y) .* log(1 .- siglog(X, theta))) + lambda/(2*m)* (theta(2:end)'*theta(2:end)); %theta(1) does not take part in computation as it's 1 - constant

    costWOReg = 1/m * sum( X .* ( siglog(X, theta) .- y ) )';

    grad = costWOReg + lambda * theta / m;

    grad(1) = costWOReg(1);

end


function g = siglog(X, theta)

    for i = 1:size(X,1)

        thetaX = theta'*X(i,:)';

        g(i) = 1 / (1 + exp(-thetaX));

    end

    g = g';

end
