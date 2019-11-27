function [J, grad] = lrCostFunction(theta, X, y, lambda)

    m = length(y);
    y1 = y.*log(sigmArr(X*theta));
    y2 = (1 .- y).*log(1 .- sigmArr(X*theta));
    reg = lambda / (2 * m) * ( theta(2:end)' * theta(2:end) );
    J = -1/m * sum( (y1 + y2) ) + reg;
    
    diff = sigmArr( X*theta ) .- y;
    grad = sum( 1/m * (diff .* X))' + lambda/m * [0; theta(2:end)];
                
end


function zArr = sigmArr(A)

    zArr = arrayfun(@sigm, A);

end


function z = sigm(x)
   
    z = 1 / (1 + exp(-x));
            
end
