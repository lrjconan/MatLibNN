function [loss_reg, diff_X, diff_reg, cov_X] = cov_off_block_diag_reg_forward(X, para)
        
lambda      = para.lambda;    
reg_mask    = para.reg_mask;
num_dim     = ndims(X);

if num_dim ~= 2 && num_dim ~= 4
    error('dimension of input data does not equal 2 or 4!');
end

if num_dim == 4
	X = X - repmat(mean(X, 4), 1, 1, 1, size(X, 4));
    X = reshape(permute(X, [3 4 1 2]), size(X, 3), size(X, 4)*size(X, 1)*size(X, 2));     
end

[dim, num]  = size(X);
mean_X      = mean(X, 2);    
diff_X      = X - repmat(mean_X, 1, num);
cov_X       = (diff_X * diff_X')./num;        
diff_reg    = cov_X - cov_X.*reg_mask;        
loss_reg    = sum(diff_reg(:).^2)*lambda/(2*dim*dim);    
