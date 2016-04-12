function [dX, dPara] = sigmoid_layer_backward(dY, X, para)

Y       = sigmoid_layer_forward(X);
dX      = (Y.*(1-Y)).*dY;
dPara   = [];