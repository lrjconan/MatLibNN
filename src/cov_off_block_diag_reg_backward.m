function [dX] = cov_off_block_diag_reg_backward(diff_X, diff_reg, para)
        
    lambda = para.lambda;
    [dim, num] = size(diff_X);   
%     dX = (diff_reg.*(lambda/(dim*dim))) * diff_X .*(2/num);

    dX = diff_reg * diff_X .*(2*lambda/(dim*dim*num));
    