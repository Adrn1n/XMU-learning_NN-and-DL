% syms x_sym
% simplify(diff(ReLU(x_sym)))
function grad=ReLU_backward(x,grad_y)
grad=grad_y.*ReLU(ReLU(x)./x);
grad(isnan(grad))=0;
end
