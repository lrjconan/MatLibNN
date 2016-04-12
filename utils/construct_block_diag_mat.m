function A = construct_block_diag_mat(m, n, b)

A = zeros(m, n);
[idxRow, idxCol] = ind2sub([m n], 1:m*n);

idxRow = idxRow(:);
idxCol = idxCol(:);

for i = 1 : numel(idxRow)
    idxRowBlock = floor((idxRow(i)-1) / b);
    idxColBlock = floor((idxCol(i)-1) / b);
    
    if idxRowBlock == idxColBlock
        A(idxRow(i), idxCol(i)) = 1;
    end
end