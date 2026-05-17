% syms x_sym
% simplify(diff(Sigmoid(x_sym)))
function grad=Sigmoid_backward(x,grad_y)
y=Sigmoid(x);
grad=grad_y.*y.*(1-y);
end
