function compress_simulations
% PURPOSE: policy search applied to diff experiments 
% organize by the results we're trying to simulate
% INPUTS:
%   x - [T x 1] vector containing the sequence of stimuli (cues) over time T
%   R - [S x A] matrix of stimulus-action reward probabilities
%               (e.g. R = [0.5 0.5] for 'uncontrollable' environment and
%               R = [0.9 0.1] for 'controllable' environment if S = 1 and A:{Go, No-Go}
%
% OUTPUTS:
%   data - structure with the following fields:
%               .a - action sequence
%               .r - reward sequence
%               add other fields...
%
% Written by: Lucy Lai

addpath('/Users/lucy/Google Drive/Harvard/Projects/rc-behav/code/GoNogo-control/')

map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows
set(0, 'DefaultLineLineWidth', 2) % first three rows

% expect theta1 to be higher value for set size = 4 compared to set size = 2 (more states = more load)
% should also predict that there is more Pavlovian Go bias for a larger set size (more policy compression)

beta = .001;
%% small set size, S = 2
R = [0.75  0.25;
    0.25 0.75]; % 3 states x 2 actions
S = repmat(randperm(2),1, 100);

data.s2 = go_nogo(S,R, beta);

%% bigger set size, S = 4
R = [0.9 0.1;
    0.75 0.25;
    0.25 0.75;
    0.1 0.9]; % 4 states x 2 actions
S = repmat(randperm(4),1, 50);
data.s4 = go_nogo(S,R, beta);

%% plot
figure; hold on;

subplot 221; hold on;
plot(data.s2.theta);
subplot 222;  hold on;
plot(data.s2.gobias,'Color',map(1,:));

subplot 221; hold on;
plot(data.s4.theta);
legend('\theta_1, set size = 2','\theta_2, set size = 2','\theta_1, set size = 4','\theta_2, set size = 4'); legend('boxoff');
xlabel('trial')
ylabel('parameter value');

subplot 222;  hold on;
plot(data.s4.gobias,'Color',map(3,:));
xlabel('trial')
ylabel('Go bias')
legend('set size = 2','set size = 4'); legend('boxoff');


subplot 223; hold on;
plot(data.s2.cost,'-')
plot(data.s4.cost,'-')
xlabel('timesteps')
ylabel('policy cost C(\pi_\theta)')
subplot 224; hold on;
plot(data.s2.rpe)
plot(data.s4.rpe)
xlabel('timesteps')
ylabel('rpe')
subprettyplot(2,2)

suptitle(strcat('\beta =',num2str(beta)))
end

function data = go_nogo(S,R,beta)

%% init
% initialize variables
nS = numel(unique(S)); nA = 2;        % #states x #actions
theta = zeros(1,nA);   % policy weights
V = zeros(nS,1);         % expected value
Q = zeros(nS,nA);        % expected value
w = zeros(nS,1);         % value weights

% learning rates
alpha_w = 0.1;          % value weight learning rate
alpha_V = 0.1;          % value learning rate
alpha_Q = 0.1;          % Q learning rate
alpha_t= 0.1;           % policy learning rate
s = 1;
num = 10;  % how many trials to asses perseveration
ecost = 0;
%beta =1;               % trade-off parameter (lower means less capacity, higher means more)

data.hitg = NaN(1,length(S)); data.hitng = NaN(1,length(S));
%% loop through stimuli
for t = 1:length(S)
    s0 = s;
    s = S(t);
    
    phi(1,:) = [V(s) Q(s,1)-Q(s,2)]; % A = Go, Q(Go - NoGo)
    phi(2,:) = [-V(s) Q(s,2)-Q(s,1)]; % A = NoGo, Q(NoGo - Go)
    mu = theta*phi';
    
    p_a = 1./(1+exp(-mu));     % action selection
    
    if rand < p_a(1)
        %s
        a = 1;   % go
    else
        %s
        a = 2;   % no-go
    end
    
    %a = (rand < p_a(1))+1;     % sample action (go: a = 1, nogo: a = 2)
    A(t) = a;
    r = rand < R(s,a);
    
    % value update
    V0 = V;
    V(s) = V(s)+alpha_V*(r-V(s));
    
    Q0 = Q;
    Q(s,a) = Q(s,a)+alpha_Q*(r-Q(s,a));
    
    
    %pa(1) = sum(A==1)/t; pa(2) = sum(A==2)/t;             % marginal action distribution
    %pa = pa+0.01; pa = pa./sum(pa);
    
    
    %     if t>num
    %         pa(1) = sum(A(t-num:t)==1)/num; pa(2) = sum(A(t-num:t)==2)/num;             % marginal action distribution
    %
    %         pa = pa+0.01; pa = pa./sum(pa);
    %     else
    pa(1) = sum(A==1)/t; pa(2) = sum(A==2)/t;             % marginal action distribution
    pa1 = pa+0.01; pa = pa./sum(pa);
    for i = 1:size(R,1)
        ps(i) = sum(S(1:t)==i)/t;
    end
    %pa(1) = sum(ps.*p_a(1));
    %pa(2) = sum(ps.*p_a(2));
    %end
    
    cost = log(p_a./pa);   %if p_a and pa are pretty similar, log(1) = 0 (low cost)
    cost = cost(a);                               % policy cost for that state-action pair
    ecost = 0.001*(cost-ecost);
    
    rpe = r + 0.98*V(s) - V0(s0) - ((1./beta)*ecost);                           % TD error
    
    % policy update
    pgrad = phi(a,:) - sum(phi.*p_a');              % policy gradient
    theta = theta + alpha_t*rpe*pgrad;         % policy weight update
    
    
    % go-bias
    if size(R,1) == 2
        if s == 1
            if a == 1 % go to win -- hit
                data.hitg(t) = 1;
            else
                data.hitg(t) = 0;
            end
        elseif s == 2
            if a == 2 % no-go to win  -- hit
                data.hitng(t) = 1;
            else
                data.hitng(t) = 0;
            end
        end
    elseif size(R,1) == 4
        if s <=2
            if a == 1 % go to win -- hit
                data.hitg(t) = 1;
            else
                data.hitg(t) = 0;
            end
        elseif s >= 3
            if a == 2 % no-go to win  -- hit
                data.hitng(t) = 1;
            else
                data.hitng(t) = 0;
            end
        end
        
    end
    
    % store results
    data.a(t) = a;
    data.r(t) = r;
    %data.w(t,:) = w;
    data.rpe(t) = rpe;
    data.theta(t,:) = theta;
    data.phi(:,:,t) = phi;
    data.rpe(t) = rpe;
    data.V(t,:) = V;
    data.Q(:,:,t) = Q;
    data.cost(t) = ecost;
    
end
data.hitg(isnan(data.hitg)) = [];
data.hitng(isnan(data.hitng)) = [];
data.gobias = movmean(movmean(data.hitg,5)-movmean(data.hitng,5),5);

end