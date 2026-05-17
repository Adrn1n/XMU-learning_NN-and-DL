%{
47, 37220232203755, 刘行逸
%}
PARAMS
[train_set,train_label]=load_data(TRAIN_F_NAME,TRAIN_LABEL_NAME);
[test_set,test_label]=load_data(TEST_F_NAME,TEST_LABEL_NAME);
train_names=fieldnames(train_set);test_names=fieldnames(test_set);
l=length(Dims);
par=cell(l-1,1);phi=cell(l-1,1);dp_r=ones(1,l-2)*ratio;phi_bw=cell(l-1,1);pre_grad=cell(l-1,1);
for i=1:(l-1)
    par{i}=randn(Dims(i)+1,Dims(i+1))*sqrt(2/Dims(i));
    pre_grad{i}=0;
    if i<(l-1)
        phi{i}=@ReLU;phi_bw{i}=@ReLU_backward;
    else
        phi{i}=@Softmax;phi_bw{i}=@Softmax_backward;
    end
end
losses=zeros(1,length(epoch));
for ep=epoch
    for i=1:length(train_names)
        if ~strcmp(train_names{i},TRAIN_LABEL_NAME)
            xs=train_set.(train_names{i});xs=transform_x(xs);
            [ys_,caches,masks]=MLP_dropout(xs,par,phi,dp_r,true);ys=train_label;
            grad=CrossEntropy_backward(ys,ys_,1,EPS);[~,grad]=MLP_dropout_backward(caches,par,dp_r,masks,phi_bw,grad);
            % [~,grad]=MLP_dropout_backward(caches,par,dp_r,masks,phi_bw,ys_-ys);
            for j=1:length(par)
                grad{j}=mean(grad{j},1:ndims(grad{j})-2);
                [par{j},pre_grad{j}]=update_mmt(par{j},reshape(grad{j},size(grad{j},2:ndims(grad{j}))),pre_grad{j},alpha,beta);
            end
        end
    end
    n=0;
    for i=1:length(test_names)
        if ~strcmp(test_names{i},TEST_LABEL_NAME)
            xs=test_set.(test_names{i});xs=transform_x(xs);
            [ys_,~,~]=MLP_dropout(xs,par,phi,dp_r,false);ys=test_label;
            losses(ep)=losses(ep)+sum(CrossEntropy(ys,ys_,EPS));n=n+size(ys_,1);
        end
    end
    losses(ep)=losses(ep)/n;
end

plot(epoch,losses)
