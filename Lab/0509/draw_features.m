function draw_features(train_features,test_features,size_train,size_train_mean,size_test,colors,marker_train,marker_train_mean,marker_test)
figure;hold on
for c=1:length(colors)
    tr=train_features{c};scatter(tr(:,1),tr(:,2),size_train,colors(c,:),marker_train);
    m=mean(tr,1);
    scatter(m(1),m(2),size_train_mean,colors(c,:),marker_train_mean);
    te=test_features{c};
    scatter(te(:,1),te(:,2),size_test,colors(c,:),marker_test)
end
hold off
end
