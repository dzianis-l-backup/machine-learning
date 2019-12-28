function A = sigmaFn(X)
    A = arrayfun(@sigmaElementFn, X);
end

function sigma = sigmaElementFn(x)
    sigma = 1/(1 + exp(-x));
end

