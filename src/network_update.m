function [W, dW_old] = network_update(W, dW, dW_old, training_para, iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update Parameters of Neural Network
%
% Input: 
%   W: parameters of neural network
%   dW: change of parameters (typically gradient)
%   dW_old: last change of parameters
%   training_para: hyper-parameters for updating, e.g. learnging
%                  rate, momentum
%   iter: iteration number of training
%
% Output:
%   W: updated parameters
%   dW_old: updated change of parameters
%
% Author:
%   Renjie Liao
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lr = training_para.layer_lr; 
momentum = training_para.layer_momentum;
weight_decay = training_para.layer_weight_decay;

% from bottom layer to top layer
for i = 1 : length(dW)            
    if isstruct(dW{i})
        fields = fieldnames(dW{i});
        num_para = length(fields);

        for j = 1 : num_para            
            W_old_tmp = W{i}.(fields{j});

            if iter == 1
                W{i}.(fields{j}) = W{i}.(fields{j}) - dW{i}.(fields{j}).*lr(i) - ...
                    W{i}.(fields{j}).*(weight_decay(i)*lr(i));                
            else
                W{i}.(fields{j}) = W{i}.(fields{j}) - dW{i}.(fields{j}).*lr(i) - ...
                    W{i}.(fields{j}).*(weight_decay(i)*lr(i)) + dW_old{i}.(fields{j}).*momentum(i);
            end

            dW_old{i}.(fields{j}) = W{i}.(fields{j}) - W_old_tmp;            
        end
    end
end
