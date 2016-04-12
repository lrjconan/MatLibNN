function [dX, dPara] = softmax_layer_backward(dY, X, para)

Y       = softmax_layer_forward(X, para);
dX      = (Y.*(1-Y)).*dY;
dPara   = [];