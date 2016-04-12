function [weight_decay] = compute_weight_decay(network_para, training_para)
    
weight_decay = 0;
weight_decay_set = {'W', 'b', 'V', 'd', 'filters', 'bias'};

% from bottom layer to top layer
for i = 1 : length(network_para)            
    if isstruct(network_para{i, 1})
        fields = fieldnames(network_para{i, 1});
        num_para = length(fields);
        
        for j = 1 : num_para
            if ismember(fields{j}, weight_decay_set)
                weight = network_para{i, 1}.(fields{j});
                weight_decay = weight_decay + training_para.layer_weight_decay(i) * sum(weight(:).^2);            
            end
        end
    end
end

weight_decay = 0.5*weight_decay;
