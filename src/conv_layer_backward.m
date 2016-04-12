function [dX, dPara] = conv_layer_backward(dY, X, para)

if ~isfield(para, 'filters')
    error('No weights specified for convolution layer!');
end

if ~isfield(para, 'pad_w') || ~isfield(para, 'pad_h')
    error('No padding specified for convolution layer!');
end

[height, width, num_channels, num_imgs] = size(X);
% [height, width, num_filters, num_imgs] = size(dY);
[filter_h, filter_w, filter_c, num_filters] = size(para.filters);

if num_channels ~= filter_c
    error('Channels of data and filters are not matched!');
end

pad_w = para.pad_w;
pad_h = para.pad_h;

X = padarray(X, [pad_h pad_w 0 0]);
kernel = reshape(para.filters, filter_h * filter_w * filter_c, num_filters);
dX = zeros(height, width, num_channels, num_imgs);
dPara.filters = zeros(size(para.filters));

if isfield(para, 'bias')
    dPara.bias = zeros(size(para.bias));
end

for n = 1 : num_imgs
    dY_tmp = reshape(dY(:, :, :, n), height * width, num_filters);
    img_patch = im2col_cube(X(:, :, :, n), [filter_h, filter_w]);
	
    if isfield(para, 'bias')
        dPara.bias = dPara.bias + sum(dY_tmp);
    end
    
    dPara.filters = dPara.filters + reshape(img_patch * dY_tmp, filter_h, filter_w, filter_c, num_filters);
        
    dX(:, :, :, n) = col2im_cube(kernel * dY_tmp', [filter_h filter_w], [height, width], [pad_h, pad_w]);
end
