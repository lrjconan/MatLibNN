function dX = euclidean_loss_backward(X, Y)

dX = (X - Y).*(2/numel(X));