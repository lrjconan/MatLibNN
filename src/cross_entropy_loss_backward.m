function dQ = cross_entropy_loss_backward(P, Q)

% dQ = -(P./(Q + eps))./numel(Q);
dQ = -(P./Q)./numel(Q);