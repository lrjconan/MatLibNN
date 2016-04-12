clear; clc;

filter_h    = 5;
filter_w    = 5;
filter_c    = 3;
filter_n    = 5;
num_imgs    = 2;
height      = 30;
width       = 30;

para.filters    = randn(filter_h, filter_w, filter_c, filter_n);
para.bias       = randn(1, filter_n);
para.pad_h      = floor(filter_h/2);
para.pad_w      = floor(filter_w/2);

X = rand(height, width, filter_c, num_imgs);

delta = 1.0e-7;

%% 
[Y, para]   = conv_layer_forward(X, para);
dY          = ones(size(Y));
[dX, dPara] = conv_layer_backward(dY, X, para);

%% derivative w.r.t. weight
[filter_h, filter_w, filter_c, filter_n] = size(para.filters);
dW_gt = zeros(filter_h, filter_w, filter_c, filter_n);
para_raw = para.filters;

[idx_h, idx_w, idx_c, idx_n] = ndgrid(1:filter_h, 1:filter_w, 1:filter_c, 1:filter_n);

for i = 1 : numel(idx_h)
    para.filters = para_raw;
    
    para.filters(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) = ...
        para_raw(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) - delta;    
    
    [Y1, ~] = conv_layer_forward(X, para);
    
    para.filters(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) = ...
        para_raw(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) + delta;
    
    [Y2, ~] = conv_layer_forward(X, para);

    dW_gt(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) = (sum(Y2(:)) - sum(Y1(:)))/(2*delta);
end

para.filters = para_raw;

%% derivative w.r.t. data
dX_gt = zeros(size(dX));
[idx_h, idx_w, idx_c, idx_n] = ndgrid(1:height, 1:width, 1:filter_c, 1:num_imgs);

for i = 1 : numel(idx_h)
    tmpX = X;
    
    tmpX(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) = ...
       X(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) - delta;    
    
    [Y1, ~] = conv_layer_forward(tmpX, para);
    
    tmpX(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) = ...
       X(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) + delta;
    
    [Y2, ~] = conv_layer_forward(tmpX, para);

    dX_gt(idx_h(i), idx_w(i), idx_c(i), idx_n(i)) = (sum(Y2(:)) - sum(Y1(:)))/(2*delta);
end

%% derivative w.r.t. bias
db_gt = zeros(size(para.bias));
para_raw = para.bias;

for i = 1 : filter_n
    para.bias = para_raw;    
    
    para.bias(i) = para_raw(i) - delta;
    
    [Y1, ~] = conv_layer_forward(X, para);
    
    para.bias(i) = para_raw(i) + delta;
    
    [Y2, ~] = conv_layer_forward(X, para);

    db_gt(i) = (sum(Y2(:)) - sum(Y1(:)))/(2*delta);
end

%%
fprintf('max diff W = %e\n', max(abs(dPara.filters(:) - dW_gt(:))));
fprintf('max diff X = %e\n', max(abs(dX(:) - dX_gt(:))));
fprintf('max diff bias = %e\n', max(abs(dPara.bias(:) - db_gt(:))));
