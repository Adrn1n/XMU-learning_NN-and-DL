function y=Softmax(x)
Exps=exp(x-max(x,[],ndims(x)));
y=Exps./sum(Exps,ndims(x));
end
