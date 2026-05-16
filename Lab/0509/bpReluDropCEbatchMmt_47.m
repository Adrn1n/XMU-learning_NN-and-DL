%{
47, 37220232203755, 刘行逸
%}
PARAMS
[data_set,label]=load_data(TRAIN_F_NAME,TRAIN_LABEL_NAME);
names=fieldnames(data_set);
l=length(Dims);
par=cell(l-1,1);phi=cell(l-1,1);phi_bw=cell(l-1,1);pre_grad=cell(l-1,1);
for i=1:(l-1)
    par{i}=rand(Dims(i)+1,Dims(i+1));
    pre_grad{i}=0;
    if i<(l-1)
        phi{i}=@ReLU;phi_bw{i}=@ReLU_backward;
    else
        phi{i}=@Softmax;phi_bw{i}=@Softmax_backward;
    end
end
dp_r=ones(1,l-2)*ratio;
for ep=epoch
    for i=1:length(names)
        if ~strcmp(names{i},TRAIN_LABEL_NAME)
            xs=data_set.(names{i});sz=size(xs);xs=reshape(xs,sz(3:ndims(xs)),[]);
            [ys_,caches,masks]=MLP_dropout(xs,par,phi,dp_r,true);ys=label;
            grad=CrossEntropy_backward(ys,ys_,1,EPS);
            [~,grad]=MLP_dropout_backward(caches,par,dp_r,masks,phi_bw,grad);
            for j=1:length(par)
                grad{j}=mean(grad{j},1:ndims(grad{j})-2);
                [par{j},pre_grad{j}]=update_mmt(par{j},reshape(grad{j},size(grad{j},2:ndims(grad{j}))),pre_grad{j},alpha,beta);
            end
        end
    end
end
