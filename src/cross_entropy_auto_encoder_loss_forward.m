function loss = cross_entropy_auto_encoder_loss_forward(P, Q)

loss = -sum(sum(P.*log(Q) + (1-P).*log(1 - Q)))./numel(Q);