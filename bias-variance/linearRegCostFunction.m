function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

  m = length(y);
  errors = (X * theta - y).^2;
  reg = theta( [2:end],1 ).^2;
  J = 1/(2*m) * sum(errors) + lambda/( 2*m ) * sum( reg );
  thetaGrad = [0; theta( [2:end],1 )];
  grad = 1/m * ( (sum( (X * theta - y) .* X ))(:) + lambda*thetaGrad );

end

