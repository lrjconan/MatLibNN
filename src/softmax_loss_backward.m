function dX = softmax_loss_backward(X, Y)

max_X   = repmat(max(X), size(X, 1), 1);
logSum  = log(repmat(sum(exp(X - max_X)), size(X, 1), 1)) + max_X;
dX      = (exp(X - logSum) - 1).*Y./numel(X);
