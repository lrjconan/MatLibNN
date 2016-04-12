function loss = cross_entropy_loss_forward(P, Q)

% loss = -sum(P(:).*log(Q(:) + eps))/numel(Q);
loss = -sum(P(:).*log(Q(:)))/numel(Q);