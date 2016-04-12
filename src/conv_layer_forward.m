function [Y, para] = conv_layer_forward(X, para)

if ~isfield(para, 'filters')
    error('No weights specified for convolution layer!');
end

if ~isfield(para, 'pad_w') || ~isfield(para, 'pad_h')
    error('No padding specified for convolution layer!');
end

[height, width, num_channels, num_imgs] = size(X);
[filter_h, filter_w, filter_c, num_filters] = size(para.filters);

if num_channels ~= filter_c
    error('Channels of data and filters are not matched!');
end

pad_w = para.pad_w;
pad_h = para.pad_h;

X = padarray(X, [pad_h pad_w 0 0]);
Y = zeros(height, width, num_filters, num_imgs);

kernel = reshape(para.filters, filter_h * filter_w * filter_c, num_filters);

for n = 1 : num_imgs
    img_patch   = im2col_cube(X(:, :, :, n), [filter_h, filter_w]);
    
    if isfield(para, 'bias')
        Y(:, :, :, n) = reshape(img_patch' * kernel + repmat(para.bias(:)', size(img_patch, 2), 1), height, width, num_filters);
    else
        Y(:, :, :, n) = reshape(img_patch' * kernel, height, width, num_filters);
    end
end
