function train(epoch,train_names,TRAIN_LABEL_NAME,train_set,par,phi,dp_r,train_label,EPS,phi_bw,pre_grad,alpha,beta,test_names,TEST_LABEL_NAME,test_set,test_label)
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
end
