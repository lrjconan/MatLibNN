function [network_forward, network_backward] = get_network_from_arch(network_arch)

num_layer           = length(network_arch);
network_forward     = cell(num_layer, 1);
network_backward    = cell(num_layer, 1);

for i = 1 : num_layer
    network_forward{i,1}    = eval(sprintf('@%s_forward', network_arch{i}));
    network_backward{i,1}   = eval(sprintf('@%s_backward', network_arch{i}));    
end

