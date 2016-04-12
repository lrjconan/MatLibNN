function [Y, para] = relu_layer_forward(X, para)

Y = X;
Y(Y < 0) = 0;