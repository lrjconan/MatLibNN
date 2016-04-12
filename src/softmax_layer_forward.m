function [Y, para] = softmax_layer_forward(X, para)

Y = exp(X);
Y = Y./repmat(sum(Y), size(Y, 1), 1);