function [acc,features]=evaluate(dataset,label,label_name,par,phi,dp_r)
names=fieldnames(dataset);
acc=0;n=0;
features=cell(size(label,2),1);
for i=1:length(names)
    if ~strcmp(names{i},label_name)
        xs=dataset.(names{i});xs=transform_x(xs);
        [ys_,caches,~]=MLP_dropout(xs,par,phi,dp_r,false);[~,cls]=max(label,[],2);
        acc=acc+sum(cls==get_predict_from_sm(ys_));
        len=size(ys_,1);
        n=n+len;
        for j=1:len
            features{cls(j)}=cat(1,features{cls(j)},caches{end-1}(j,1:2));
        end
    end
end
acc=acc/n;
end
