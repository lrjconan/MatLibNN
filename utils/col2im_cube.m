function [img] = col2im_cube(img_patch, sz_patch, sz_img, pad)

[dim_patch, num_patch] = size(img_patch);

if num_patch ~= (sz_img(1) + 2 * pad(1) - sz_patch(1) + 1) * (sz_img(2) + 2 * pad(2) - sz_patch(2) + 1);
    error('number of patches is wrong!');
end

stride = sz_patch(1) * sz_patch(2);
num_channels = dim_patch / stride;
img = zeros(sz_img(1), sz_img(2), num_channels);

for i = 1 : num_channels	            
    tmp = col2im_2D(img_patch((i-1)*stride + 1 : i*stride, :), sz_patch, sz_img+2*pad, false);    
    img(:, :, i) = tmp(pad(1) + 1 : pad(1) + sz_img(1), pad(2) + 1 : pad(2) + sz_img(2));
end
