function data = load_data(dataset)

% Load data sets.
%
% USAGE: data = load_data(dataset)
%
% INPUTS:
%   dataset - 'collins18' or 'collins14'

switch dataset
    
    case 'collins18'
        
        T = {'ID' 'learningblock' 'trial' 'ns' 'state' 'iter' 'corchoice' 'action' 'reward' 'rt' 'pcor' 'delay' 'phase'};
        x = csvread('Collins18_data.csv',1);
        S = unique(x(:,1));
        for s = 1:length(S)
            ix = x(:,1)==S(s) & x(:,end)==0;
            for j = 1:length(T)
                data(s).(T{j}) = x(ix,j);
            end
        end
        
    case 'collins14'
        
        load Collins_JN2014.mat
        
        T = {'ID' 'learningblock' 'ns' 'trial' 'state' 'image' 'folder' 'iter' 'corchoice' 'action' 'key' 'cor' 'reward' 'rt' 'cond' 'pcor' 'delay'};
        S = unique(expe_data(:,1));
        for s = 1:length(S)
            ix = expe_data(:,1)==S(s);
            for j = 1:length(T)
                data(s).(T{j}) = expe_data(ix,j);
            end
            data(s).ID = data(s).ID(1);
            data(s).cond = data(s).cond(1);
            data(s).C = 3;
            data(s).N = length(data(s).learningblock);
        end
        
end
end