function results = analyze_collins14(data)

% Construct reward-complexity curve for Collins et al. (2014) data.

if nargin < 1
    data = load_data('collins14');
end

beta = linspace(0.1,10,50);
beta = logspace(log10(0.1),log10(10),50);

% run Blahut-Arimoto
for s = 1:length(data)
    B = unique(data(s).learningblock);
    setsize = zeros(length(B),1);
    R_data =zeros(length(B),1);
    V_data =zeros(length(B),1);
    for b = 1:length(B)
        ix = data(s).learningblock==B(b);
        state = data(s).state(ix);
        c = data(s).corchoice(ix);
        action = data(s).action(ix);
        r = data(s).reward(ix);
        err(b) = sum(action ~= c)/length(action);
        
        R_data(b) = mutual_information(state,action,0.1);
        V_data(b) = mean(data(s).reward(ix));
        
        S = unique(state);
        Q = zeros(length(S),3);
        Ps = zeros(1,length(S));
        
        for a = 1:max(action)
            Pa(a) = mean(action==a); % marginal action probability
        end
        
        for i = 1:length(S)
            ii = state==S(i);
            Ps(i) = mean(ii);
            a = c(ii); a = a(1);
            Q(i,a) = 1;
            
            for a = 1:max(action)
                Pas(i,a) = mean(action(state==i)==a); % conditioned on state
            end
            
            % errors
            sta = state(state==i);
            act = action(state==i);
            cor = c(state==i);
            rew = r(state==i);
            err_s(b,i) = sum(act~=cor)/length(act); % by block AND state
            V_data_s(i) = mean(rew);
        end
        R_data_s = nansum(Pas.*log(Pas./Pa),2)'; % KL divergence for each state
        
        [R(b,:),V(b,:)] = blahut_arimoto(Ps,Q,beta);
        
        % compute bias (subj x stimulus x set size)
        Vd3 =  interp1(R(b,:),V(b,:),R_data(b));
        bias_s(b,1:length(S)) = (Vd3 - V_data_s); % bias is per subject x stim x set size (85 x 6 x 5)

        
        setsize(b) = length(S)-1;
        
        clear Pa Pas 
        clear V_data_s R_data_s
    end
    
    for c = 1:max(setsize)
        results.R(s,:,c) = nanmean(R(setsize==c,:));
        results.V(s,:,c) = nanmean(V(setsize==c,:));
        results.R_data(s,c) = nanmean(R_data(setsize==c));
        results.V_data(s,c) = nanmean(V_data(setsize==c));
        results.err(s,c) = nanmean(err(setsize==c));
        results.err_s(s,:,c) = squeeze(nanmean(err_s(setsize==c,:)));
        results.bias_s(s,:,c) = squeeze(nanmean(bias_s(setsize==c,:)));
    end
    
     clear R V
     clear bias_s err_s
    
end

% compute bias (subj x set size)
R = squeeze(nanmean(results.R)); % avg over subjects
V = squeeze(nanmean(results.V));
for c = 1:max(setsize)
    Vd2(:,c) =  interp1(R(:,c),V(:,c),results.R_data(:,c));
    results.bias(:,c) = Vd2(:,c) - results.V_data(:,c); % bias is per subject x set size (85 x 5)
    results.V_interp(:,c) = Vd2(:,c);
end

% prettystuff
figure; hold on;
subplot 221; hold on;
p = plot(mean(R'),mean(V'),'LineWidth',4);
n = size(beta,2);
cd = [uint8(brewermap(n,'Blues')*255) uint8(ones(n,1))].';
drawnow
set(p.Edge, 'ColorBinding','interpolated', 'ColorData',cd)

xlabel('Policy complexity')
ylabel('Average reward')
prettyplot
axis square
subplot 222;
p = plot(beta,mean(R'),'-','LineWidth',4)
drawnow
set(p.Edge, 'ColorBinding','interpolated', 'ColorData',cd)
xlabel('\beta')
ylabel('Policy complexity')
prettyplot
axis square
subplot 223;
p = plot(beta,mean(V'),'-','LineWidth',4)
drawnow
set(p.Edge, 'ColorBinding','interpolated', 'ColorData',cd)
xlabel('\beta')
ylabel('Average reward')
prettyplot
axis square


% fit empirical reward-complexity curves with polynomial
cond = [data.cond];
for j = 1:2
    results.bic = zeros(max(setsize),2);
    results.aic = zeros(max(setsize),2);
    for c = 1:max(setsize)
        x = results.R_data(cond==j-1,c);
        x = [ones(size(x)) x x.^2];
        y = results.V_data(cond==j-1,c);
        n = length(y); k = size(x,2);
        [b,bint] = regress(y,x);
        results.bci_sep(c,j,:) = diff(bint,[],2)/2;
        results.b_sep(c,j,:) = b;
        mse = mean((y-x*b).^2);
        results.bic(c,1) = results.bic(c,1) + n*log(mse) + k*log(n);
        results.aic(c,1) = results.bic(c,1) + n*log(mse) + k*2;
        
        x = results.R_data(:,c);
        x = [ones(size(x)) x x.^2];
        y = results.V_data(:,c);
        n = length(y); k = size(x,2);
        [b,bint] = regress(y,x);
        results.bci_joint(c,j,:) = diff(bint,[],2)/2;
        results.b_joint(c,j,:) = b;
        mse = mean((y-x*b).^2);
        results.bic(c,2) = n*log(mse) + k*log(n);
        results.aic(c,2) = n*log(mse) + k*2;
        
    end
end
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