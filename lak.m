function sim_lak
%% lak
% simulate decision making trials with conditions L [100:0, 80:20, 65:35, 55:45, 45:55, 35:65, 20:80 and 0:100] R,
% stimuli features are the decision confidences
% sort choices by their features

% run a simulation, and show you can reproduce the result that conditioning
% on the

% model can explain choice history bias
%

% show how the RPE should relate to confidence and cost
agent.lrate_V = 0.1;
agent.lrate_p = 0.1;
agent.lrate_theta = 0.05;
agent.sigma = 0.2;
beta = [0.5 1 2 5 10]; % capacity constraint
beta = 2;

% state "centers" mixture probabilities
b = [0, 20, 35, 45, 55, 65, 80, 100]*0.01;

s = 1:8; % discretized indices of stimuli
s = repmat(s, 1, 500);  % 500 repeats of each stim
s = s(randperm(length(s)));

% correct choice
c = ones(size(s));
c(s>4) = 2;
% B [100:0, 80:20, 65:35, 55:45, 45:55, 35:65, 20:80 and 0:100] A

data.state = s;
data.belief = b;                 % belief centers (means)
data.corchoice = c;

figure; hold on;

for c = 1:length(beta) 
    agent.beta = beta(c);
    for i = 1:60 % run i agents
        simdata = sim_lak(agent,data);
        for x = 1:max(data.state)                        % for each states
            
            % nansummary choice probabilities
            A = simdata.action(data.state==x);  % sort into 8 x 100 matrix
            fracA(i,x) = nansum(A==2)/length(A);   % choose (A)
        
            % conditioned on previous action
            simdata.prevA = [NaN simdata.action(1:end-1)];         % previous action vector
            AA = simdata.action(simdata.prevA==2 & data.state==x); % previous choice = A
            AB = simdata.action(simdata.prevA==1 & data.state==x); % previous choice = B
            fracAA(i,x) = nansum(AA==2)/length(AA);  % choose (A) when prev choice = A
            fracAB(i,x) = nansum(AB==2)/length(AB);  % choose (A) when prev choice = B
            
            % conditioned on previous stimulus
            simdata.prevO = [NaN simdata.state(1:end-1)];                  % previous odor vector
            for prevx = 1:max(data.state)                                  % for each previous odor
                AX = simdata.action(simdata.prevO==prevx & data.state==x); % previous odor = prevx & current odor = x
                fracAX(prevx,x,i) = nansum(AX==2)/length(AX);                 % choose (A) when previous odor = prevx
            end
            
        end
        
    end  % i agents
    
    % fraction choice(A) vs. odor
    subplot 221; hold on;
    errorbar(b,mean(fracA),sem(fracA,1),'ko-','LineWidth',2)
    
    % frac choice(A)t sorted by choice(t-1)
    subplot 222; hold on;
    errorbar(b,nanmean(fracAA),sem(fracAA,1),'bo-','LineWidth',2)
    errorbar(b,nanmean(fracAB),sem(fracAB,1),'go-','LineWidth',2)
    
    % frac choice(A)t sorted by odor(t-1)
    subplot 223; hold on;
    c = winter(6);
    c = [c(4:6,:);c(1:3,:)];
    for i = 7:-1:2 % prevO
        %errorbar(repmat(b,8,1)',nanmean(fracAX,3)',sem(fracAX,3)','o-','LineWidth',2)
        errorbar(b,nanmean(fracAX(i,:,:),3),sem(fracAX(i,:,:),3),'o-','LineWidth',2,'Color',c(i-1,:))
    end
    % plot(b,mean(fracA),'o-')
end   % beta

subplot 221;
prettyplot
axis square; axis([0 1 0 1])
ylabel('Fraction choice A')
xlabel('Current odor (%A)')
%lg = legend(string(beta),'Location','Southeast');
%title(lg,'\beta')
%legend('boxoff')

subplot 222;
prettyplot
axis square; axis([0 1 0 1])
ylabel('Fraction choice A')
xlabel('Current odor (%A)')
lg = legend('A','B','Location','Southeast');
title(lg,'Previous choice')
legend('boxoff')

subplot 223;
prettyplot
axis square; axis([0 1 0 1])
ylabel('Fraction choice A')
xlabel('Current odor (%A)')
lg = legend(string(b(7:-1:2)),'Location','Southeast');
title(lg,'Prev. odor')
legend('boxoff')


suptitle(strcat('\beta =',num2str(beta)));
why
% frac choice(A)t sorted by choice(t-1)


%% simulate collins with high and low beta (capacity)

rng(1); % set random seed for reproducibility

if nargin < 1
    data = load_data;
end

if nargin < 2
    load model_fits;
end

s = 10;
x = results(1).x(s,:);
agent.lrate_theta = x(2);
agent.lrate_V = 0.01;%x(3);
agent.lrate_p = 0.03;
agent.beta = .05;
agent.nc = 0;
simdata1 = actor_critic2(agent,data(s));

agent.nc = 1; % no cost
simdata_nc1 = actor_critic2(agent,data(s));

% higher beta
agent.nc = 0;
agent.beta = 10;
simdata2 = actor_critic2(agent,data(s));

agent.nc = 1; % no cost
simdata_nc2 = actor_critic2(agent,data(s));

%% plots


figure; hold on;
idx = simdata1.setsize==2;
ss = (simdata1.state(idx)==1);
idx(1:500) = 0;

subplot 211; plot(simdata1.action(idx),'k.-','MarkerSize', 20);prettyplot; % beta = low; pret
title('\beta = 1')
subplot 212; plot(simdata2.action(idx),'r.-','MarkerSize', 20);prettyplot; % beta = high
title('\beta = 5')
%plot(simdata_nc1.action(idx),'b.-','MarkerSize', 20)
%plot(simdata_nc2.action(idx),'m.-','MarkerSize', 20)

figure; hold on;
plot(simdata1.cost(idx), simdata1.rpe(idx),'k.','MarkerSize', 20) % beta = low
plot(simdata2.cost(idx), simdata2.rpe(idx),'r.','MarkerSize', 20) % beta = high
plot(simdata_nc1.cost(idx), simdata_nc1.rpe(idx),'b.','MarkerSize', 20)
plot(simdata_nc2.cost(idx), simdata_nc2.rpe(idx),'m.','MarkerSize', 20)
xlabel('cost')
ylabel('\delta')
prettyplot


figure; hold on;
plot(simdata1.cost, simdata1.rpe,'k.','MarkerSize', 20) % beta = low
plot(simdata2.cost, simdata2.rpe,'r.','MarkerSize', 20) % beta = high
plot(simdata_nc1.cost, simdata_nc1.rpe,'b.','MarkerSize', 20)
plot(simdata_nc2.cost, simdata_nc2.rpe,'m.','MarkerSize', 20)
xlabel('Policy cost')
ylabel('RPE \delta')
prettyplot

figure; hold on;
plot(simdata1.cost, simdata1.rpe,'r.','MarkerSize', 20) % beta = low
plot(simdata_nc1.cost, simdata_nc1.rpe,'k.','MarkerSize', 20)
xlabel('Policy cost')
ylabel('RPE \delta')
legend('cost-sensitive agent', 'regular TD learning')
legend('boxoff')
prettyplot

end

function simdata = sim_lak(agent, data)

simdata = data;
state = data.state;
b = data.belief;
sigma = agent.sigma; % noise
corchoice = data.corchoice;          % correct choice on each trial
setsize = length(unique(state));     % number of distinct stimuli
theta = zeros(setsize,2);            % policy parameters (8 states, 2 actions)
w = zeros(setsize,1);                % state value weights
p = ones(1,2)/2;                     % marginal action probabilities

for t = 1:length(state)                  % loop through states
    s = state(t);                        % stimulus idx
    bt = normrnd(b(s),sigma);            % draw belief from N(s,sigma), s^{hat}
    phi = exp(-(bt-b).^2./sigma.^2)';    % phi - basis function
    
    d = agent.beta*(theta'*phi)' + log(p);
    logpolicy = d - logsumexp(d);
    policy = exp(logpolicy);    % softmax policy
    a = fastrandsample(policy); % action
    if a == corchoice(t)        % a = 1 for L and a = 2 for R
        r = 1;                  % reward
    else
        r = 0;
    end
    cost = logpolicy(a) - log(p(a));                          % policy complexity cost
    
    V = w'*phi;
    rpe = agent.beta*r - cost - V;                            % reward prediction error
    g = agent.beta*phi*(1 - policy(a));                       % policy gradient
    theta(:,a) = theta(:,a) + (agent.lrate_theta)*rpe*g;    % policy parameter update  /t
    
    
    w = w + agent.lrate_V*rpe*phi;
    
    p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
    simdata.action(t) = a;
    simdata.reward(t) = r;
    %simdata.value(t) = V;
end

end