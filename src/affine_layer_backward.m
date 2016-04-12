function [dX, dPara] = affine_layer_backward(dY, X, para)

num_dim_X = ndims(X);

if num_dim_X ~= 4 && num_dim_X ~= 2
    error('Dimension of X should 2 or 4!');
end

h = 1; w = 1; c = 1; n = 1;

if num_dim_X == 4
    [h, w, c, n] = size(X);
    X = reshape(X, h*w*c, n);
else
    [c, n] = size(X);
end

% Y = W * X + b
if isfield(para, 'W')
    dX = para.W' * dY;
    
    if isfield(para, 'diag')
        dPara.W = diag(diag(dY * X'));
    else
        dPara.W = dY * X';
    end
    
    if isfield(para, 'b')
        dPara.b = sum(dY, 2);
    end
else
    error('No weights specified for affine layer!');
end    

if num_dim_X == 4
    dX = reshape(dX, h, w, c, n);
end