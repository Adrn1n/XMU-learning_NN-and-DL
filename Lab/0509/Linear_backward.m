function [grad_x,grad_W]=Linear_backward(W,x_with_b,grad_y)
sz=size(W);
grad_x=pagemtimes(grad_y,reshape(W,[sz(1:end-2),sz(end),sz(end-1)]));
nd=ndims(grad_x);
idx=repmat({':'},1,nd);
idx{end}=1:size(grad_x,nd)-1;
grad_x=grad_x(idx{:});
sz=size(grad_y);
% grad_W=pagemtimes(reshape(x_with_b,[size(x_with_b),1]),reshape(grad_y,[sz(1:end-1),1,sz(end)]));
grad_W=reshape(x_with_b,[size(x_with_b),1]).*reshape(grad_y,[sz(1:end-1),1,sz(end)]);
end
