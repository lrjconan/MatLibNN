function [corr_mat] = compute_pearson_corr(X)

% calculate the Pearson coefficient as affinity matrix
% generalize the Pearson coefficient to 4D as RV coefficient

num_dim = ndims(X);

if num_dim ~= 2 && num_dim ~= 4
    error('dimension of input data does not equal 2 or 4!');
end

X_centered = zeros(size(X));

if num_dim == 2
    X_centered  = X - repmat(mean(X, 2), 1, size(X, 2));
else
	X_centered  = X - repmat(mean(X, 4), 1, 1, 1, size(X, 4));
    X_centered  = reshape(permute(X_centered, [3 4 1 2]), size(X, 3), size(X, 4)*size(X, 1)*size(X, 2));
end

X_centered  = X_centered./repmat(sqrt(sum(X_centered.^2, 2)), 1, size(X_centered, 2));
corr_mat    = X_centered * X_centered';
