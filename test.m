function test
data = load_data('collins18'); % just get the stimuli
s = 10; % random subject stimulus presentation
B = unique(data(s).learningblock);
ix = data(s).learningblock==B(1);
state = data(s).state(ix);
corchoice = data(s).corchoice(ix);
max(state)
nSubj = length(state); % number of subjects to simulate
nErrs = 1:length(state); % number of errors
rcol = plmColors(nSubj,'r');
set(0, 'DefaultAxesColorOrder',colormap(rcol))

for a = 1:3
    Pa(a) = mean(corchoice==a); % marginal action probability
end

for i = 1:nSubj
   fakedata(i).state = state; 
   fakedata(i).corchoice = corchoice;
   fakedata(i).action = corchoice;
   fakedata(i).action(nErrs(i):end) = randp(Pa,[length(nErrs)-nErrs(i)+1 1]);  
end 

% subject with high complexity, on the curve but off curve

fakedata(1).action = corchoice;                    % subject with high complexity, on the curve
%fakedata(2).action = randp(Pa,size(state));        % subject with low complexity, on the curve

%fakedata(3).action = randp(Pa,size(state));         % subject with high complexity, but off the curve
%fakedata(4).action = randp(Pa,size(state));         % subject with low complexity, but off curve

data = fakedata;
results = analyze_rc(data);


%% plotting
ylim = [0.25 1.1];
xlim = [0 0.9];

figure; hold on; prettyplot; 
subplot 121; hold on; 
for s = 1:length(results)
    plot(results(s).R_data, results(s).V_data,'.','MarkerSize',50);
end
plot(results(1).R,results(1).V,'k','LineWidth',3);
xlabel('Policy complexity');
ylabel('Average reward');
set(gca,'YLim',ylim,'XLim',xlim);

% errors for each state
subplot 122; hold on;
for i = 1:length(data) % number of subjects
    for s = 1:3 % 3 states
        a = data(i).action(data(i).state==s);
        c = data(i).corchoice(data(i).state==s);
        errors(i,s) = sum(a~=c)/length(a);
    end
    plot(errors(i), results(i).bias,'.','MarkerSize',50); % errors vs bias away from the curve
     
end
%subplot 122; hold on;
%plot(errors, [results.bias],'.','MarkerSize',50); % errors vs bias away from the curve
xlabel('% Errors');
ylabel('Bias');

end

function results = analyze_rc(data)
for s = 1:length(data)
    beta = linspace(0.1,15,50);
    state = data(s).state;
    c = data(s).corchoice;
    action = data(s).action;
    
    results(s).R_data = mutual_information(state,action,0.01);
    results(s).V_data = mean(action==c);
    
    S = unique(state);
    Q = zeros(length(S),3);
    Ps = zeros(1,length(S));
    for i = 1:length(S)
        ii = state==S(i);
        Ps(i) = mean(ii);
        a = c(ii); a = a(1);
        Q(i,a) = 1;
        
        for a = 1:3
            Pa(a) = mean(action==a); % marginal action probability
            Pas(i,a) = mean(action(state==i)==a); % conditioned on state
        end
    end
    
    [R,V] = blahut_arimoto(Ps,Q,beta);
    
    KL = nansum(Pas.*log(Pas./Pa),2); % KL divergence between policies for each state
    
    results(s).KL = KL;
    results(s).Pa = Pa;
    results(s).Pas = Pas;
    results(s).R = R;
    results(s).V = V;
 
    Vd2 =  interp1(R,V,results(s).R_data);
    results(s).bias = Vd2 - results(s).V_data;
  


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
