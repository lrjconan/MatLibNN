clear; 
% clc;
rng('default');
rng(1);

num_imgs    = 10;
num_filters = 512;
img_size    = 256;
filter_size = 11;
num_channels= 3;
pad         = floor(filter_size/2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mask1 = fspecial('gaussian', filter_size, 1.0);
kernel1 = randn(filter_size, filter_size, num_channels, num_filters);
img1    = rand(img_size, img_size, num_channels, num_imgs);
C1      = zeros(img_size, img_size, num_filters, num_imgs);

tic;
for n = 1 : num_imgs
    for i = 1 : num_filters
        tmp = zeros(img_size, img_size);
        for j = 1 : num_channels        
            tmp = tmp + imfilter(img1(:, :, j, n), kernel1(:, :, j, i));
        end
        
        C1(:, :, i, n) = tmp;
    end
end
time1 = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% C2 = zeros(img_size, img_size, num_filters, num_imgs);
% tic;
% for n = 1 : num_imgs
%     for c = 1 : num_channels
%         kernel2     = shiftdim(reshape(kernel1(:, :, c, :), filter_size * filter_size, num_filters), 1);
%         img2        = padarray(img1(:, :, c, n), [pad pad]);
%         img_patch   = im2col(img2, [filter_size, filter_size], 'sliding');
%         img_res     = kernel2 * img_patch;
%         
%         for i = 1 : num_filters 
%             C2(:, :, i, n) = C2(:, :, i, n) + col2im(img_res(i, :)', [1 1], [img_size img_size], 'sliding');
%         end
%         
% %         C2(:, :, :, n) = C2(:, :, :, n) + shiftdim(reshape(kernel2 * img_patch, num_filters, img_size, img_size), 1);
%     end
% end
% time2 = toc;

C2 = zeros(img_size, img_size, num_filters, num_imgs);
kernel2 = reshape(kernel1, filter_size * filter_size * num_channels, num_filters)';

tic;
img2    = padarray(img1, [pad pad 0 0]);
for n = 1 : num_imgs        
    img_patch   = im2col_cube(img2(:, :, :, n), [filter_size, filter_size]);    
    img_res     = kernel2 * img_patch;
%     C2(:, :, :, n) = shiftdim(reshape(img_res, num_filters, img_size, img_size), 1);
    C2(:, :, :, n) = reshape(img_res', img_size, img_size, num_filters);
end
time2 = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
img2 = zeros(img_size + 2*pad);
img2(pad + 1 : pad + img_size, pad + 1 : pad + img_size) = img1;
imgs = repmat(img2(:), 1, num_filters);
tmp = zeros(img_size + 2*pad);
tmp(1:filter_size, 1:filter_size) = mask1;
% kernel = toeplitz(sparse(tmp(:)'));

tic;
kernel = toeplitz([tmp(1); zeros((img_size + 2*pad)^2-1, 1)], tmp(:)');
idx = [];
for i = 1 : img_size-1
    idx = [idx; (i*(img_size + 2*pad) - filter_size + 2 : i*(img_size + 2*pad))'];
end
idx = [idx; (size(kernel, 1) - ((filter_size-1)*(img_size+2*pad) + filter_size) + 2 : size(kernel, 1))'];
kernel(idx, :) = [];

C2 = kernel * imgs;
time2 = toc;
%}

fprintf('Time 1 = %f\n', time1);
fprintf('Time 2 = %f\n', time2);
fprintf('Diff = %f\n', norm(C1(:) - C2(:)));