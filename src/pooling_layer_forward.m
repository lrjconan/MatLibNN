function [Y, para] = pooling_layer_forward(X, para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% currently only support pooling for 2D image or 4D tensor
%
% para.method = {'avg', 'max'}
% para.kernel_h =  
% para.kernel_w = 
% para.subsample_w =    apply subsampling on the response map before pooling 
% para.subsample_h = 
% para.stride = stide for pooling
% para.pad_size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_dims = ndims(X);

if num_dims ~= 2 && num_dims ~= 4
    error('currently only support pooling for 2D image or 4D tensor');
end

field_names = {'kernel_w', 'kernel_h', 'subsample_w', 'subsample_h', ...
    'stride', 'pad_size', 'method'};

for i = 1 : numel(field_names)
    if ~isfield(para, field_names{i})
        error('Field "%s" missing in structure "para"!', field_names{i});
    end
end

kernel_w    = para.kernel_w;
kernel_h    = para.kernel_h;
subsample_w = para.subsample_w;
subsample_h = para.subsample_h;
stride      = para.stride;
pad_size    = para.pad_size;

%% calculate the output size
[size_in_h, size_in_w, size_in_c, size_in_n] = size(X);
size_out_h = floor(( size_in_h + 2 * pad_size - ((kernel_h - 1) * subsample_h + 1) ) / stride) + 1;
size_out_w = floor(( size_in_w + 2 * pad_size - ((kernel_w - 1) * subsample_w + 1) ) / stride) + 1;
[h_out, w_out] = ndgrid(1 : size_out_h, 1 : size_out_w);
h_out = h_out(:);
w_out = w_out(:);

%%
patch_h = kernel_h * subsample_h;   % window size for pooling
patch_w = kernel_w * subsample_w;
X = padarray(X, [pad_size, pad_size, 0, 0]);
Y = zeros(size_out_h, size_out_w, size_in_c, size_in_n);

if strcmp(para.method, 'max')    
    for n = 1 : size_in_n
        for c = 1 : size_in_c
            for i = 1 : numel(h_out)                                  
                h_in_start  = (h_out(i) - 1) * stride + 1; 
                w_in_start  = (w_out(i) - 1) * stride + 1;                               
                h_in_end    = min(size(X, 1), h_in_start + patch_h - 1);
                w_in_end    = min(size(X, 2), w_in_start + patch_w - 1);                
                
                pool_window = X(h_in_start : h_in_end, w_in_start : w_in_end, c, n);                
                pool_mask   = zeros(size(pool_window));
                pool_mask(1 : subsample_h : end, 1 : subsample_w : end) = 1;
                pool_window = pool_window.*pool_mask;                
                
                Y(h_out(i), w_out(i), c, n) = max(pool_window(:));
            end
        end    
    end
elseif strcmp(para.method, 'avg')
    for n = 1 : size_in_n
        for c = 1 : size_in_c
            for i = 1 : numel(h_out)                                  
                h_in_start  = (h_out(i) - 1) * stride + 1; 
                w_in_start  = (w_out(i) - 1) * stride + 1;                               
                h_in_end    = min(size(X, 1), h_in_start + patch_h - 1);
                w_in_end    = min(size(X, 2), w_in_start + patch_w - 1);                
                
                pool_window = X(h_in_start : h_in_end, w_in_start : w_in_end, c, n);                
                pool_mask   = zeros(size(pool_window));
                pool_mask(1 : subsample_h : end, 1 : subsample_w : end) = 1;
                pool_window = pool_window.*pool_mask;                
                
                Y(h_out(i), w_out(i), c, n) = mean(pool_window(:));
            end
        end    
    end    
else
    error('Not supported method!');
end
