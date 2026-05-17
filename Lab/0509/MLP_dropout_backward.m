function [grad_x,grad_param]=MLP_dropout_backward(cache,param,ratio,mask,phi_bw,grad_y)
grad_param=cell(length(param),1);
for i=1:length(param)
    if i>1
        grad_y=dropout_backward(ratio(end+2-i),mask{end+2-i},grad_y);
        % grad_y=phi_bw{end+1-i}(cache{end+2-2*i},grad_y);
    end
    grad_y=phi_bw{end+1-i}(cache{end+2-2*i},grad_y);
    [grad_y,grad_param{end+1-i}]=Linear_backward(param{end+1-i},cache{end+1-2*i},grad_y);
end
grad_x=grad_y;
end
