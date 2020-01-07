function v = fromClassToVector(y, num_labels) % decimal y into array of 0,1
    v = zeros(num_labels, 1);
    v(y) = 1;
end
