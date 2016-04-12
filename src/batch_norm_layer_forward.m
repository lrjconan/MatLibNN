function [Y, para] = batch_norm_layer_forward(X, para)

batch_size  = size(X, 2);

if ~isfield(para, 'mean')
    % for the training stage
    mean_X = repmat(mean(X, 2), 1, batch_size);
else
    % for the testing stage    
    mean_X = repmat(para.mean, 1, batch_size);
end

if ~isfield(para, 'std')
    % for the training stage
    std_X = sqrt(var(X, 1, 2) + para.BN_epsilon);
else
    % for the testing stage    
    std_X = para.std;
end

cen_X       = X - mean_X;
Y           = cen_X./repmat(std_X, 1, batch_size);
para.cen_X  = cen_X;
para.std_X  = std_X;
