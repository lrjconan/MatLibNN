function loss = softmax_loss_forward(X, Y)

max_X   = repmat(max(X), size(X, 1), 1);
logSum  = log(repmat(sum(exp(X - max_X)), size(X, 1), 1)) + max_X;
loss    = -sum(sum(Y.*(X - logSum)))./numel(X);
