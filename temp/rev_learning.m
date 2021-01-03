function rev_learning
%% reversal learning

agent.lrate_V = 0.8;
agent.lrate_p = 0.1;
agent.lrate_theta = 0.8;
beta = [0.5 1 2 5 10]; % capacity constraint
beta = 0.5;
stimrep = 100;
tpb = 20;
rev = stimrep*2/tpb; % number of reversals  = repeats / trials per block

s = 1:2;                    % discretized indices of stimuli
s = repmat(s, 1, stimrep);  % stim repeats, there will be rep x 2 trials
s = s(randperm(length(s)));

data.state = s;
data.R1 = [.8 .2; .2 .8];
data.R2 = [.2 .8; .8 .2];

data.R1 = [1 -1; -1 1];
data.R2 = [-1 1; 1 -1];
data.reversal = [];

for i = 1:rev
    data.reversal = [data.reversal ones(1,tpb)+mod(i,2)];
end


for c = 1:length(beta) 
    agent.beta = beta(c);
    for i = 1:10 % run i agents
        simdata = sim(agent,data);
        pCorr = [reshape(simdata.corchoice(data.reversal==1),tpb,rev/2)'; reshape(simdata.corchoice(data.reversal==2),tpb,rev/2)'];
        %pCorr2 = reshape(simdata.corchoice(data.reversal==2),tpb,rev/2)';
    end  % i agents
    
    % pCorr after reversal
    subplot 211; hold on;
    errorbar(1:tpb,mean(pCorr),sem(pCorr,1),'ko-','LineWidth',2)
    %errorbar(1:tpb,mean(pCorr2),sem(pCorr2,1),'ko-','LineWidth',2)
    axis([0 tpb 0 1])
    prettyplot
    
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


end

function simdata = sim(agent, data)

simdata = data;
state = data.state;
R1 = data.R1;
R2 = data.R2;
setsize = length(unique(state));     % number of distinct stimuli
theta = zeros(setsize,2);            % policy parameters (8 states, 2 actions)
V = zeros(setsize,1);                % state value weights
p = ones(1,2)/2;                     % marginal action probabilities

for t = 1:length(state)                  % loop through states
    s = state(t);                        % stimulus idx
    d = agent.beta*theta(s,:) + log(p);
    logpolicy = d - logsumexp(d);
    policy = exp(logpolicy);    % softmax policy
    a = fastrandsample(policy); % action
    if data.reversal(t) == 1
        r = rand < R1(s,a);           % reward
        simdata.corchoice(t) = s==a;
    else
    	r = rand < R2(s,a);           % reward  
        simdata.corchoice(t) = s~=a;
    end
    
    cost = logpolicy(a) - log(p(a));                          % policy complexity cost
    
    rpe = agent.beta*r - V(s)- cost;                         % reward prediction error
    g = agent.beta*(1 - policy(a));                           % policy gradient
    theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;      % policy parameter update
    V(s) = V(s) + agent.lrate_V*rpe;  
    
    p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
    simdata.action(t) = a;
    simdata.reward(t) = r;
    %simdata.value(t) = V;
end

end