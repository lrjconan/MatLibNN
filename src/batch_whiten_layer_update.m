function [para] = batch_whiten_layer_update(X, para)

W = para.V * para.U;
b = para.d - W * para.c;

num_X   = size(X, 2);    
para.c  = mean(X, 2);
diff_X  = X - repmat(para.c, 1, num_X);
cov_X   = (diff_X * diff_X')./num_X;    

[E, S, ~] = svd(cov_X);
tmp_S   = sqrt(diag(S) + para.BW_epsilon);

para.U  = diag((tmp_S).^(-1)) * E';
para.V  = W * (E * diag(tmp_S));
para.d  = b + W * para.c;
