function loss = euclidean_loss_forward(X, Y)

loss = sum((X(:) - Y(:)).^2)./numel(X);