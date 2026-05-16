function [param,grad]=update_mmt(param,grad,pre_grad,alpha,beta)
grad=grad+beta*pre_grad;
param=param-alpha*grad;
end
