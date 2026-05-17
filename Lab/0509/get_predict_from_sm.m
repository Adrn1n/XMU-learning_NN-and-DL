function pred=get_predict_from_sm(sm)
nd=ndims(sm);
mask=(sm==max(sm,[],nd));
ids=cell(1,nd);
[ids{:}]=ind2sub(size(sm,nd:-1:1),find(permute(mask,nd:-1:1)));
ids=ids{1};
cnts=sum(mask,nd);
if nd<3
    offsets=rand(size(sm,1),1)*size(sm,nd);
else
    offsets=rand(size(sm,1:nd-1))*size(sm,nd);
end
offsets=mod(floor(offsets),cnts);
cnts=permute(cnts,nd:-1:1);cnts=cnts(:);starts=[0;cumsum(cnts(1:end-1))];
pred=ids(reshape(starts,size(sm,1:nd-1),[])+offsets+1);
end
