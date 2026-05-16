% dim=input("dim = ");x_sym=sym("x_sym",[1,dim]);y_sym=sym("y_sym",[dim,1]);
% simplify(jacobian(Softmax(x_sym),x_sym)*y_sym)
function grad=Softmax_backward(x,grad_y)
y=Softmax(x);
grad=y.*(grad_y-sum(grad_y.*y,ndims(x)));
end
