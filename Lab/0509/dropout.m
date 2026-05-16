function [y,mask]=dropout(ratio,x)
mask=rand(size(x))>ratio;
y=x.*mask/(1-ratio);
end
