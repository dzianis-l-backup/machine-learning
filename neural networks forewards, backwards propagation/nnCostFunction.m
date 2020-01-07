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

    X = [ones(m,1), X]; % ones added to each of the training examples as the 1 neuron

    for t=1:m
        a1 = X(t,:);
        a1 = a1';

        [a3, a2] = forewards( a1, Theta1, Theta2 ); % output signals computation
        yt = transformY(y(t,1), num_labels);
        J = J + costClasses(yt, a3);
        
        [D1, D2] = backwards( yt, a3, Theta2, Theta2_grad, a2, Theta1, Theta1_grad, a1 );
        Theta1_grad = Theta1_grad + D1;
        Theta2_grad = Theta2_grad + D2;

    end

    J = J/m + lambda/(2*m) * regulize(Theta1, Theta2);
    Theta1_grad = 1/m * Theta1_grad + 1/m * regulizeGrad(Theta1, lambda);
    Theta2_grad = 1/m * Theta2_grad + 1/m * regulizeGrad(Theta2, lambda);

    % Unroll gradients
    grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

function [a3, a2] = forewards(x, theta1, theta2) %output signals computation
    z2 = theta1*x;
    a2 = g(z2);
    a2 = [ones(1,1); a2];

    z3 = theta2*a2;
    a3 = g(z3);
end
                     
function [D1, D2] = backwards(y, a3, theta2, D2, a2, theta1, D1, a1 )
    d3 = a3 - y;

    D2 = d3 * a2';
    d2 = removeFirstElem( theta2' * d3 .* a2 .* (1 .- a2) );

    D1 = d2 * a1';
end

function sigma = _f(x) %sigma function of as single variable
    sigma = 1/(1 + exp(-x));
end

function z = g(x) %sigma of a vector
    z = arrayfun(@_f, x);
end

function cost = costClasses(y,a) % cost function of all classes
    cost = -y.*log(a) - (1 .- y).*(log(1 .- a));
    cost = sum(cost);
end

function yv = transformY(y, num_labels) % decimal y into array of 0,1 transform
    yv = zeros(num_labels, 1);
    yv(y) = 1;
end

function theta = regulize(theta1, theta2)
    t1 = removeFirstColumn(theta1.*theta1);
    t2 = removeFirstColumn(theta2.*theta2);
    theta = sum(sum(t1)) + sum(sum(t2));
end

function z = zDerivative(A)
    z = g(A).*(1 .- g(A));
end

function A = removeFirstColumn(X)
    A = X(:, 2:end);
end
                         
function x = removeFirstElem(a)
    x = a(2:end);
end

function s = sizeDebug(X)
    s = size(X)
end

function reg = regulizeGrad(theta, lambda)
    reg = [ zeros( size( theta, 1 ), 1 ), lambda * removeFirstColumn(theta)  ];
end
