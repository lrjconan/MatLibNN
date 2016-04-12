function [dX, dPara] = batch_norm_layer_backward(dY, X, para)

num     = size(X, 2);
d_mean  = sum(dY, 2)./(-para.std_X);
d_var   = diag(diag(dY * para.cen_X')) * ((para.std_X.^(-3)).*(-0.5));
dX      = dY./repmat(para.std_X, 1, num) + para.cen_X.*repmat(d_var.*(2/num), 1, num) + repmat(d_mean./num, 1, num);
dPara   = [];