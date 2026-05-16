function y=CrossEntropy(x,x_sm_,eps)
y=-sum(x.*log(x_sm_+eps),ndims(x));
end
