clear; clc;

n = 10;
d = 100;
X = randn(d, n);
% Y = zeros(d, n);
% for i = 1 : n
%     Y(randi(d), i) = 1;
% end
Y = rand(d, n);
Y = Y./repmat(sum(Y), size(Y, 1), 1);

delta = 1.0e-4;

loss = softmax_loss_forward(X, Y);
dX = softmax_loss_backward(X, Y);

dX_gt = zeros(size(dX));

for i = 1 : size(X,1)    
    for j = 1 : size(X,2)

        X1 = X;
        X1(i,j) = X1(i,j) - delta;
        loss_1 = softmax_loss_forward(X1, Y);

        X2 = X;
        X2(i,j) = X2(i,j) + delta;
        loss_2 = softmax_loss_forward(X2, Y);
        
        dX_gt(i, j) = (loss_2 - loss_1)/(2*delta);
    end
end

fprintf('max diff = %e\n', max(abs(dX(:) - dX_gt(:))));
