function g = sigmoid(z)

    g = arrayfun(@sigmoidScalar, z)

end

function y = sigmoidScalar(k)

    y = 1 / (1 + exp(-k));

end
