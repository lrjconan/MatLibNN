function [loss_reg, diff_X, diff_reg, cov_X] = cov_F_reg_forward(X, para)
        
    lambda = para.lambda;
    [dim, num] = size(X);
    
    mean_X      = mean(X, 2);    
    diff_X      = X - repmat(mean_X, 1, num);
    cov_X       = (diff_X * diff_X')./num;
    diff_reg    = cov_X - eye(dim);
    loss_reg    = sum(diff_reg(:).^2)*lambda/(2*dim*dim);