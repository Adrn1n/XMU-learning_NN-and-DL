function [dataset,label]=load_data(f_name,label_name)
dataset=load(f_name);
label=dataset.(label_name);
end
