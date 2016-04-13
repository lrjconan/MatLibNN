function [network_forward, network_backward] = get_network_from_arch(network_arch)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Build Neural Network 
%
% Input: 
%   network_arch: L x 1, cell array of layer names
%
% Output:
%   network_forward: L x 1, cell array of function handles for forward
%                    computation
%   network_backward: L x 1, cell array of function handles for backward
%                    computation
%
% Author:
%   Renjie Liao
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_layer           = length(network_arch);
network_forward     = cell(num_layer, 1);
network_backward    = cell(num_layer, 1);

for i = 1 : num_layer
    network_forward{i,1}    = eval(sprintf('@%s_forward', network_arch{i}));
    network_backward{i,1}   = eval(sprintf('@%s_backward', network_arch{i}));    
end

