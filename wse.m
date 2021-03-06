function [se, m] = wse(X,dim)
    
    % Within-subject error, following method of Cousineau (2005).
    %
    % USAGE: [se, m] = wse(X,dim)
    %
    % INPUTS:
    %   X - [N x D] data with N observations and D subjects
    %   dim (optional) - dimension along which to compute within-subject
    %   variance (default: 2)
    %
    % OUTPUTS:
    %   se - [1 x D] within-subject standard errors
    %   m - [1 x D] means
    %
    % Sam Gershman, June 2015
    
    if nargin < 2; dim = 2; end
    m = squeeze(nanmean(X));
    X = bsxfun(@minus,X,nanmean(X,dim));
    N = sum(~isnan(X));
    se = bsxfun(@rdivide,nanstd(X),sqrt(N));
    se = squeeze(se);