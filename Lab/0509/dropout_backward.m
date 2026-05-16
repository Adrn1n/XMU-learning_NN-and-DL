function grad_x=dropout_backward(ratio,mask,grad_y)
grad_x=grad_y.*mask/(1-ratio);
end
