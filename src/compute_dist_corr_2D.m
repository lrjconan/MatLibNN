function [corr_mat] = compute_dist_corr_2D(X)

% calculate the distance correlation coefficient as affinity matrix

num_dim = ndims(X);

if num_dim ~= 2
    error('dimension of input data does not equal 2 or 4!');
end

[dim, num] = size(X);
corr_mat = zeros(dim);

[idx_a, idx_b] = ind2sub([dim dim], 1:dim^2);
idx_a = idx_a(:); idx_b = idx_b(:);

for i = 1 : numel(idx_a)
    if idx_a(i) >= idx_b(i)
        A = squareform(pdist(X(idx_a(i), :)'));
        B = squareform(pdist(X(idx_b(i), :)'));        
        A = A - repmat(mean(A), num, 1) - repmat(mean(A, 2), 1, num) + mean(mean(A));
        B = B - repmat(mean(B), num, 1) - repmat(mean(B, 2), 1, num) + mean(mean(B));

        dist_cov = sqrt(sum(sum(A.*B)))/num;
        dist_var_A = sqrt(sum(sum(A.*A)))/num;
        dist_var_B = sqrt(sum(sum(B.*B)))/num;

        corr_mat(idx_a(i), idx_b(i)) = dist_cov/sqrt(dist_var_A*dist_var_B);
    end
end

corr_mat = corr_mat + corr_mat' - diag(diag(corr_mat));