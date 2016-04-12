function [img] = col2im_2D(img_patch, sz_patch, sz_img, flag_overlap)
%
% col2im_2D: convert sliding patches to image 
%
img     = zeros(sz_img);
coeff   = zeros(sz_img);

num_pixel = prod(sz_patch);
[yy, xx] = ind2sub(sz_patch, 1:num_pixel);

for p = 1 : num_pixel        
    img(yy(p) : yy(p)+sz_img(1)-sz_patch(1),xx(p) : xx(p)+sz_img(2)-sz_patch(2)) = ...
        img(yy(p) : yy(p)+sz_img(1)-sz_patch(1),xx(p) : xx(p)+sz_img(2)-sz_patch(2)) + ...
        col2im(img_patch(p, :), sz_patch, sz_img, 'sliding');
    
    coeff(yy(p) : yy(p)+sz_img(1)-sz_patch(1),xx(p) : xx(p)+sz_img(2)-sz_patch(2)) = ...
        coeff(yy(p) : yy(p)+sz_img(1)-sz_patch(1),xx(p) : xx(p)+sz_img(2)-sz_patch(2)) + 1;        
end

if flag_overlap == true
    img = img ./ coeff;    
end
  