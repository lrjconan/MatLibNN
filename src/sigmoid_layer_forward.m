function [Y, para] = sigmoid_layer_forward(X, para)

Y = 1./(1 + exp(-X));