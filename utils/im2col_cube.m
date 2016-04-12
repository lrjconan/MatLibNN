function [img_patch] = im2col_cube(img, sz)

[height, width, num_channels] = size(img);
num_patch = (height - sz(1) + 1) * (width - sz(2) + 1);
stride = sz(1) * sz(2);
img_patch = zeros(stride * num_channels, num_patch);

for i = 1 : num_channels    
    img_patch((i-1)*stride + 1 : i*stride, :) = im2col(img(:, :, i), [sz(1) sz(2)], 'sliding');
end
