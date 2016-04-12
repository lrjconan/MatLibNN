function [dX] = pearson_corr_reg_backward(X, corr_mat, para)

num_dim = ndims(X);

if num_dim ~= 2 && num_dim ~= 4
    error('dimension of input data does not equal 2 or 4!');
end

dX = zeros(size(X));

if num_dim == 2    
    X_centered = X - repmat(mean(X, 2), 1, size(X, 2));    
    A = X_centered * X_centered';
    B = sum(X_centered.^2, 2);
    
    [idx_i, idx_j] = ind2sub(size(corr_mat), 1 : numel(corr_mat));
    idx_i = idx_i(:); idx_j = idx_j(:);
    
    for m = 1 : numel(idx_i)
        i = idx_i(m);
        j = idx_j(m);
        
        if i < j
            tmp = zeros(size(X));
            tmp(i, :) = corr_mat(i, j)^2 * ( X_centered(j, :)./A(i, j) - X_centered(i, :)./B(i) );
            tmp(j, :) = corr_mat(i, j)^2 * ( X_centered(i, :)./A(i, j) - X_centered(j, :)./B(j) );
            dX = dX + tmp;
        end                
    end
    
    dX = dX.*2;
else
    X_centered = X - repmat(mean(X, 4), 1, 1, 1, size(X, 4));
    X_centered = reshape(permute(X_centered, [3 4 1 2]), size(X, 3), size(X, 4)*size(X, 1)*size(X, 2));    
    A = X_centered * X_centered';
    B = sum(X_centered.^2, 2);
    
    [idx_i, idx_j] = ind2sub(size(corr_mat), 1 : numel(corr_mat));
    idx_i = idx_i(:); idx_j = idx_j(:);
    
    for m = 1 : numel(idx_i)
        i = idx_i(m);
        j = idx_j(m);
        
        if i < j
            tmp = zeros(size(X_centered));
            tmp(i, :) = corr_mat(i, j)^2 * ( X_centered(j, :)./A(i, j) - X_centered(i, :)./B(i) );
            tmp(j, :) = corr_mat(i, j)^2 * ( X_centered(i, :)./A(i, j) - X_centered(j, :)./B(j) );
            dX = dX + permute(reshape(tmp, size(X, 3), size(X, 4), size(X, 1), size(X, 2)), [3 4 1 2]);
        end
    end
    
    dX = dX.*2;
end
