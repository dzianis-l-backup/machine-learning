function p = predict(Theta1, Theta2, X)

    m = size(X, 1);
    a1 = [ones(m,1) X];
    num_labels = size(Theta2, 1);
    p = zeros(size(a1, 1), 1);
    z2 = a1 * Theta1';
    a2 = [ones(size(z2, 1),1) sigm(z2)];
    z3 = a2 * Theta2';
    a3 = sigm(z3);
    [values, indexes] = max(a3,[],2);
    p = indexes;

end

function A = sigm(X)

    A = arrayfun(@g, X);

end
                            
function z = g(x)
    
    z = 1 / (1 + exp(-x));
                            
end
