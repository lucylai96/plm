function [R,V,Pa] = blahut_arimoto(Ps,Q,b)
    
    % Blahut-Arimoto algorithm applied to the reward-complexity trade-off.
    %
    % USAGE: [R,V,Pa] = blahut_arimoto(Ps,Q,[b])
    %
    % INPUTS:
    %   Ps - [1 x S] state probabilities, where S is the number of states
    %   Q - [S x A] expected reward, where A is the number of actions
    %   b (optional) - vector of trade-off parameters. Default: linspace(0.1,15,30)
    %
    % OUTPUTS:
    %   R - [K x 1] channel capacity values, where K is the length of b
    %   V - [K x 1] average reward values
    %   Pa - [K x A] marginal action policy
    %
    % Sam Gershman, Jan 2020
    
    A = size(Q,2);
    nIter = 50;
    if nargin < 3; b = linspace(0.1,15,30); end
    R = zeros(length(b),1); V = zeros(length(b),1); Pa = zeros(length(b),A);
    
    for j = 1:length(b)
        F = b(j).*Q;
        v0 = mean(Q(:));
        q = ones(1,A)./A;
        for i = 1:nIter
            logP = log(q) + F;
            Z = logsumexp(logP,2);
            Psa = exp(logP - Z);
            q = Ps*Psa;
            v = sum(Ps*(Psa.*Q));
            if abs(v-v0) < 0.001; break; else v0 = v; end
        end
        Pa(j,:) = q;
        V(j) = v;
        R(j) = b(j)*v - Ps*Z;
    end