function [Y, para] = batch_whiten_layer_forward(X, para)

batch_size = size(X, 2);
Y = para.U * (X - repmat(para.c, 1, batch_size));
Y = para.V * Y + repmat(para.d, 1, batch_size);
