clear; clc;

n = 10;
d = 100;
W = rand(d, d).*1.0e-0;
X = rand(d, n);

delta = 1.0e-7;
para.lambda = 1.0e-0;

[loss_reg, diff_X, diff_reg, ~] = cov_off_diag_reg_forward(W*X, para);
dY = cov_off_diag_reg_backward(diff_X, diff_reg, para);
dW = dY*X';

dW_gt = zeros(size(dW));

for i = 1 : size(W,1)    
    for j = 1 : size(W,2)

        W1 = W;
        W1(i,j) = W1(i,j) - delta;
        [loss_reg_1, ~, ~, ~] = cov_off_diag_reg_forward(W1*X, para);

        W2 = W;
        W2(i,j) = W2(i,j) + delta;
        [loss_reg_2, ~, ~, ~] = cov_off_diag_reg_forward(W2*X, para);
        
        dW_gt(i, j) = (loss_reg_2 - loss_reg_1)/(2*delta);                              
    end
end

fprintf('max diff = %e\n', max(abs(dW(:) - dW_gt(:))));
