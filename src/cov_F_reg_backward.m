function [dX] = cov_F_reg_backward(diff_X, diff_reg, para)
        
    lambda = para.lambda;
    [dim, num] = size(diff_X);   
    dX = (diff_reg.*(lambda/(dim*dim))) * diff_X .*(2/num);
