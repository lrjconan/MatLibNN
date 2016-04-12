function [dW_old] = clear_momentum(dW_old)
    
for i = 1 : length(dW_old)            
    if isstruct(dW_old{i})
        fields = fieldnames(dW_old{i});
        num_para = length(fields);

        for j = 1 : num_para            
            dW_old{i}.(fields{j}) = zeros(size(dW_old{i}.(fields{j})));            
        end
    end
end
