function [par,phi,phi_bw,pre_grad]=build_NN(dims,act_func,act_func_bw)
l=length(dims);
par=cell(l-1,1);phi=cell(l-1,1);phi_bw=cell(l-1,1);pre_grad=cell(l-1,1);
for i=1:(l-1)
    par{i}=randn(dims(i)+1,dims(i+1))*sqrt(2/dims(i));
    pre_grad{i}=0;
    if i<(l-1)
        phi{i}=act_func;phi_bw{i}=act_func_bw;
    else
        phi{i}=@Softmax;phi_bw{i}=@Softmax_backward;
    end
end
end
