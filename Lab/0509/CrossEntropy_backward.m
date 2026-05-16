% dim=input("dim = ");x_sym=sym("x_sym",[1,dim]);x_sm_sym_=sym("x_sm_sym_",[1,dim]);
% simplify(gradient(CrossEntropy(x_sym,x_sm_sym_),x_sm_sym_))
function grad=CrossEntropy_backward(x,x_sm_,grad_y,eps)
grad=-grad_y.*(x./(x_sm_+eps));
end
