function p = predict(theta, X)

m = size(X, 1); % Number of training examples

p = 1 / 1 + exp(X*theta);

for i=1:m
   p(i) = 1 / (1 + exp(-X(i,:)*theta)) >= 0.5;
end

end
