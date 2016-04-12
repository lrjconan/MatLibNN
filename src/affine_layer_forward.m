function [Y, para] = affine_layer_forward(X, para)

num_dim_X = ndims(X);

if num_dim_X ~= 4 && num_dim_X ~= 2
    error('Dimension of X should 2 or 4!');
end

if num_dim_X == 4
    [h, w, c, n] = size(X);
    X = reshape(X, h*w*c, n);
end

if isfield(para, 'W')
    if isfield(para, 'b')
        Y = para.W * X + repmat(para.b, 1, size(X, 2));                
    else
        Y = para.W * X;
    end
else
    error('No weights specified for affine layer!');
end