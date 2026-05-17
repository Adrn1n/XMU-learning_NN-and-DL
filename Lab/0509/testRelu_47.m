%{
47, 37220232203755, 刘行逸
%}
bpReluDropCEbatchMmt_47
[acc,train_features]=evaluate(train_set,train_label,TRAIN_LABEL_NAME,par,phi,dp_r);
disp(acc);
[acc,test_features]=evaluate(test_set,test_label,TEST_LABEL_NAME,par,phi,dp_r);
disp(acc);
draw_features(train_features,test_features,DRAW_SIZE_TRAIN,DRAW_SIZE_TRAIN_MEAN,DRAW_SIZE_TEST,COLORS,MARKER_TRAIN,MARKER_TRAIN_MEAN,MARKER_TEST)
