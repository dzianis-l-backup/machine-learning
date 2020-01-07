function g = sigmoidGradient(z)
    g = arrayfun(@elementSigmGrad, z);
end

function g = elementSigmGrad(elem)
    g = sigm(elem)*(1 - sigm(elem));
end

function g = sigm(x)
    g = 1/(1 + exp(-x));
end
