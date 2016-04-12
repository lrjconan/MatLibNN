clear; clc;

num = 10;
dim = 100;
X = rand(dim, num);
delta = 1.0e-7;

para.U = eye(dim);
para.V = eye(dim);
para.c = zeros(dim, 1);
para.d = rand(dim, 1);
para.update_samples = X;
para.BW_epsilon = 1.0e-2;
para.iter = 1;
para.update_iter = 10;

[Y, para] = BatchWhiten_layer_forward(X, para);

% loss function = sum(Z(:))
dY = ones(size(Y));
[dX, dPara] = BatchWhiten_layer_backward(dY, X, para);

dX_gt = zeros(size(dX));
for i = 1 : size(X,1)    
    for j = 1 : size(X,2)
        X1 = X;
        X1(i,j) = X1(i,j) - delta;        
        [Y1, para] = BatchWhiten_layer_forward(X1, para);

        X2 = X;
        X2(i,j) = X2(i,j) + delta;        
        [Y2, para] = BatchWhiten_layer_forward(X2, para);
        
        dX_gt(i, j) = (sum(Y2(:)) - sum(Y1(:)))/(2*delta);               
    end
end

dV_gt = zeros(size(para.V));
V = para.V;
for i = 1 : size(para.V,1)    
    for j = 1 : size(para.V,2)
        V1 = V;
        V1(i,j) = V1(i,j) - delta;        
        para.V  = V1;
        [Y1, para] = BatchWhiten_layer_forward(X, para);

        V2 = V;
        V2(i,j) = V2(i,j) + delta;        
        para.V  = V2;
        [Y2, para] = BatchWhiten_layer_forward(X, para);
        
        dV_gt(i, j) = (sum(Y2(:)) - sum(Y1(:)))/(2*delta);
    end
end

dd_gt = zeros(size(para.d));
d = para.d;
for i = 1 : length(para.d)    
    d1 = d;
    d1(i) = d1(i) - delta;        
    para.d  = d1;
    [Y1, para] = BatchWhiten_layer_forward(X, para);

    d2 = d;
    d2(i) = d2(i) + delta;
    para.d  = d2;
    [Y2, para] = BatchWhiten_layer_forward(X, para);

    dd_gt(i) = (sum(Y2(:)) - sum(Y1(:)))/(2*delta);    
end

fprintf('max diff = %e\num', max(abs([dX(:); dPara.V(:); dPara.d(:)] - [dX_gt(:); dV_gt(:); dd_gt(:)])));

