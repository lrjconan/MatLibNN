function dQ = cross_entropy_auto_encoder_loss_backward(P, Q)

dQ = (-P./(Q + eps) + (1 - P)./(1 - Q + eps))./numel(Q);