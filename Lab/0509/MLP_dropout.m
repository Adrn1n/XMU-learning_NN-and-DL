function [y,cache,mask]=MLP_dropout(x,param,phi,ratio,is_train)
cache=cell(2*length(param),1);mask=cell(length(param)-1,1);
for i=1:length(param)
    sz=size(x);
    cache{2*i-1}=cat(ndims(x),x,ones(sz(1:end-1),1));cache{2*i}=Linear(param{i},cache{2*i-1});x=phi{i}(cache{2*i});
    if is_train && i<length(param)
        [x,mask{i}]=dropout(ratio(i),x);
    end
end
y=x;
end
