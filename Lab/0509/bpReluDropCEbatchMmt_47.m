%{
47, 37220232203755, 刘行逸
%}
PARAMS
[train_set,train_label]=load_data(TRAIN_F_NAME,TRAIN_LABEL_NAME);
[test_set,test_label]=load_data(TEST_F_NAME,TEST_LABEL_NAME);
train_names=fieldnames(train_set);test_names=fieldnames(test_set);
[par,phi,phi_bw,pre_grad]=build_NN(DIMS,@ReLU,@ReLU_backward);
dp_r=ones(1,length(DIMS)-2)*ratio;
train(epoch,train_names,TRAIN_LABEL_NAME,train_set,par,phi,dp_r,train_label,EPS,phi_bw,pre_grad,alpha,beta,test_names,TEST_LABEL_NAME,test_set,test_label);
