function [J grad] = nnCostFunction(nn_params,
                                   input_layer_size,
                                   hidden_layer_size,
                                   num_labels,
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)),
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end),
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1), X]; % ones added to each of the training examples as the 1 neuron

for i=1:m

    a = forewards( X(i,:)', Theta1, Theta2 ); % output signals computation
    J = J + costClasses(transform(y(i,1), num_labels), a);

end

J = J/m;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function a = forewards(x, theta1, theta2) %output signals computation
    z2 = theta1*x;
    a2 = g(z2);
    a2 = [ones(1,1); a2];

    z3 = theta2*a2;
    a = g(z3);

end

function sigma = f(x) %sigma function of as single variable
    sigma = 1/(1 + exp(-x));
end

function z = g(x) %sigma of a vector
    z = arrayfun(@f, x);
end

function cost = costClasses(y,a) % cost function of all classes
    cost = -y.*log(a) - (1 .- y).*(log(1 .- a));
    cost = sum(cost);
end

function yv = transform(y, num_labels) % decimal y into array of 0,1 transform
    yv = zeros(num_labels, 1);
    yv(y) = 1;
end
