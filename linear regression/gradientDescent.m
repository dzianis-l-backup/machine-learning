function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1); % initial cost function vector

    for iter = 1:num_iters

        features = length(theta); %number of features
        thetaCopy = theta; %copying the theta

        for j = 1:length(theta)
            thetaCopy(j) = thetaCopy(j) - alpha * 1/m * sum ( ( X * theta - y ) .* X(:, j) );
        end

    theta = thetaCopy;

    J_history(iter) = computeCost(X, y, theta);

    end

end
