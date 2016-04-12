function [loss, corr_mat] = pearson_corr_reg_forward(X, para)

corr_mat = compute_pearson_corr(X);
loss = 0.5*norm(corr_mat - diag(diag(corr_mat)), 'fro')^2;