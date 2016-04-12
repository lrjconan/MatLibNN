clear; clc;

n = 10;
d = 100;
X = rand(d, n);

delta = 1.0e-7;
para.BN_epsilon = 1.0e-3;

[Y, para] = BatchNorm_layer_forward(X, para);

% loss function = sum(Z(:))
dY = ones(size(Y));
[dX, dPara] = BatchNorm_layer_backward(dY, X, para);

dX_gt = zeros(size(dX));

for i = 1 : size(X,1)
    for j = 1 : size(X,2)
        X1 = X;
        X1(i,j) = X1(i,j) - delta;        
        [Y1, para] = BatchNorm_layer_forward(X1, para);

        X2 = X;
        X2(i,j) = X2(i,j) + delta;        
        [Y2, para] = BatchNorm_layer_forward(X2, para);
        
        dX_gt(i, j) = (sum(Y2(:)) - sum(Y1(:)))/(2*delta);               
    end
end

fprintf('max diff = %e\n', max(abs(dX(:) - dX_gt(:))));

