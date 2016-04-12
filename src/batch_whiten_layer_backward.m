function [dX, dPara] = batch_whiten_layer_backward(dY, X, para)

dX = (para.V * para.U)' * dY;
dPara.V = dY * (para.U * (X - repmat(para.c, 1, size(X, 2))))';
dPara.d = sum(dY, 2);
