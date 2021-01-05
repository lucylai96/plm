function results = analyze_collins(data)

% Analyze Collins (2018) data.

if nargin < 1
    data = load_data('collins18');
end

beta = linspace(0.1,15,50);

for s = 1:length(data)
    B = unique(data(s).learningblock);
    cond = zeros(length(B),1);
    R_data =zeros(length(B),1);
    V_data =zeros(length(B),1);
    for b = 1:length(B)
        ix = data(s).learningblock==B(b) & data(s).phase==0;
        state = data(s).state(ix);
        c = data(s).corchoice(ix);
        action = data(s).action(ix);
        action(action==-1) = 2;
        for a = 1:max(action)
            Pa(a) = mean(action==a); % marginal action probability
        end
        R_data(b) = mutual_information(state,action,0.1);
        V_data(b) = mean(data(s).reward(ix));
        
        S = unique(state);
        Q = zeros(length(S),3);
        Ps = zeros(1,length(S));
        for i = 1:length(S)
            ii = state==S(i);
            Ps(i) = mean(ii);
            a = c(ii); a = a(1);
            Q(i,a) = 1;
            
            for a = 1:max(action)
                Pas(i,a) = mean(action(state==i)==a); % conditioned on state
            end
        end
        
        KL = nansum(Pas.*log(Pas./Pa),2); % KL divergence between policies for each state
        C(b) = sum(Ps'.*KL);
        
        
        [R(b,:),V(b,:)] = blahut_arimoto(Ps,Q,beta);
        
        if length(S)==3
            cond(b) = 1;
        else
            cond(b) = 2;
        end
        
        clear Pa Pas
    end
    % figure; plot(C',R_data,'ko'); prettyplot; axis square; dline;
    % xlabel('empirically computed I(S;A)')
    % ylabel('matlab function estimated I(S;A)')
    
    %R_data = C';
    
    for c = 1:2
        results.R(s,:,c) = nanmean(R(cond==c,:));
        results.V(s,:,c) = nanmean(V(cond==c,:));
        results.R_data(s,c) = nanmean(R_data(cond==c));
        results.V_data(s,c) = nanmean(V_data(cond==c));
        %results.Pa(:,:) = Pa;
        
        if results.R_data(s,c) < 0.15 % tag an example subject low complexity
            results.ex(s,c,1) = s;   % results.R_data(s,c);
            results.ex(s,c,2) = results.R_data(s,c);
        end
        
        if results.R_data(s,c) > 0.48 % tag an example subject high complexity
            results.ex(s,c,1) = s;
            results.ex(s,c,2) = results.R_data(s,c);
        end
    end
    
    clear R V
    
end

p = signrank(results.R_data(:,1),results.R_data(:,2))

R = squeeze(nanmean(results.R));
V = squeeze(nanmean(results.V));
for c = 1:2
    Vd2(:,c) =  interp1(R(:,c),V(:,c),results.R_data(:,c));
    results.bias(:,c) = results.V_data(:,c) - Vd2(:,c);
end

[r,p] = corr([results.V_data(:,1); results.V_data(:,2)],[Vd2(:,1); Vd2(:,2)])
[r,p] = corr([results.R_data(:,1); results.R_data(:,2)],abs([results.bias(:,1); results.bias(:,2)]))

end

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
end

function I = mutual_information(state,action,alpha)
    
    % Hutter estimator of mutual information.
    %
    % USAGE: I = mutual_information(state,action,[alpha])
    
    uS = unique(state);
    uA = unique(action);
    
    N = zeros(length(uS),length(uA));
    if nargin < 3; alpha = 1/numel(N); end % Perks (1947) prior
    
    for x = 1:length(uS)
        for y = 1:length(uA)
            N(x,y) = alpha + sum(state==uS(x) & action==uA(y));
        end
    end
    
    n = sum(N(:));
    nA = sum(N);
    nS = sum(N,2);
    P = psi(N+1) - psi(nA+1) - psi(nS+1) + psi(n+1);
    I = sum(sum(N.*P))/n;
end