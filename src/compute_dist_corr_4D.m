function [corr_mat_spatial, corr_mat_channel] = compute_dist_corr_4D(X)

% calculate the distance correlation coefficient as affinity matrix

num_dim = ndims(X);

if num_dim ~= 4
    error('dimension of input data does not equal 2 or 4!');
end

[height, width, num_channel, num] = size(X);
map_size = height * width;
corr_mat_spatial = zeros(map_size);
corr_mat_channel = zeros(num_channel);

%% calculate spatial correlation
[idx_a, idx_b] = ind2sub([map_size map_size], 1:map_size^2);
idx_a = idx_a(:); idx_b = idx_b(:);

for i = 1 : numel(idx_a)
    if idx_a(i) >= idx_b(i)
        hA = mod(idx_a(i)-1, height) + 1; wA = floor((idx_a(i) - 1)/height) + 1;
        hB = mod(idx_b(i)-1, height) + 1; wB = floor((idx_b(i) - 1)/height) + 1;
        A = squareform(pdist(squeeze(X(hA, wA, :, :))'));
        B = squareform(pdist(squeeze(X(hB, wB, :, :))'));
        A = A - repmat(mean(A), num, 1) - repmat(mean(A, 2), 1, num) + mean(mean(A));
        B = B - repmat(mean(B), num, 1) - repmat(mean(B, 2), 1, num) + mean(mean(B));

        dist_cov = sqrt(sum(sum(A.*B)))/num;
        dist_var_A = sqrt(sum(sum(A.*A)))/num;
        dist_var_B = sqrt(sum(sum(B.*B)))/num;

        corr_mat_spatial(idx_a(i), idx_b(i)) = dist_cov/sqrt(dist_var_A*dist_var_B);
    end
end

corr_mat_spatial = corr_mat_spatial + corr_mat_spatial' - diag(diag(corr_mat_spatial));

%% calculate channel correlation
[idx_a, idx_b] = ind2sub([num_channel num_channel], 1:num_channel^2);
idx_a = idx_a(:); idx_b = idx_b(:);

for i = 1 : numel(idx_a)
    if idx_a(i) >= idx_b(i)
        A = squareform(pdist(reshape(X(:, :, idx_a(i), :), map_size, num)'));
        B = squareform(pdist(reshape(X(:, :, idx_b(i), :), map_size, num)'));
        A = A - repmat(mean(A), num, 1) - repmat(mean(A, 2), 1, num) + mean(mean(A));
        B = B - repmat(mean(B), num, 1) - repmat(mean(B, 2), 1, num) + mean(mean(B));

        dist_cov = sqrt(sum(sum(A.*B)))/num;
        dist_var_A = sqrt(sum(sum(A.*A)))/num;
        dist_var_B = sqrt(sum(sum(B.*B)))/num;

        corr_mat_channel(idx_a(i), idx_b(i)) = dist_cov/sqrt(dist_var_A*dist_var_B);
    end
end

corr_mat_channel = corr_mat_channel + corr_mat_channel' - diag(diag(corr_mat_channel));
