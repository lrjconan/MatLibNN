function [mask] = construct_cov_mask_from_label(label)

num = length(label);
mask = zeros(num);
val = unique(label);

for i = 1 : numel(val)
    idx = label == val(i);
    mask(idx, idx) = 1;
end
