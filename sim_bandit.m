function simdata = sim_bandit(agent)

rng(1)
% contextual bandits

nS = 3;   % # state features
nA = 2;   % # actions
theta = zeros(nS,nA);                 % policy parameters (13 state-features, 4 actions)
V = zeros(nS,1);                      % state value weights
p = ones(1,nA)/nA;                    % marginal action probabilities
nTrials = 50;
% A B, L R
%R = [0.8 0.8; 0.2 0.2];
R = [1 0; 1 0];
%R = [1 0; 0 1];
state = 1:2;                     % discretized indices of stimuli
state = repmat(state, 1, nTrials);   % stim repeats, there will be rep x 2 trials
%state = state(randperm(length(state)));

% 100 trials
for t = 1:length(state)
    s = state(t);
    
    phi = zeros(nS,1);
    phi(s) = 1;
    phi(3) = 1;
    
    % policy
    d = agent.beta*(theta'*phi)' + log(p);
    logpolicy = d - logsumexp(d);
    policy = exp(logpolicy);    % softmax policy
    a = fastrandsample(policy); % action
    
    %r = rand < R(s,a);           % reward
    r = R(s,a);
    
    cost = logpolicy(a) - log(p(a));                        % policy complexity cost
    
    % learning updates
    rpe = agent.beta*r - cost - V(s);                       % reward prediction error
    g = agent.beta*phi*(1 - policy(a));                         % policy gradient
    theta(:,a) = theta(:,a) + (agent.lrate_theta)*rpe*g;    % policy parameter update
    
    V = V + agent.lrate_V*rpe*phi;
    
    p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
    simdata.action(t) = a;
    simdata.reward(t) = r;
    simdata.state(t) = s;
    
end
simdata.theta = theta;
simdata.pa = p;

phi = [1 0 1;
    0 1 1]';
for i = 1:2
    d = agent.beta*(theta'*phi(:,i))' + log(p);
    logpolicy = d - logsumexp(d,2);
    policy = exp(logpolicy);    % softmax
    simdata.pas(i,:) = policy;
end

simdata.KL = nansum(simdata.pas.*log(simdata.pas./simdata.pa),2); % KL div for each state (row), before and after reversal (column)
simdata.V = [sum(simdata.state ==1 & simdata.reward ==1) sum(simdata.state==2 & simdata.reward==1)]/nTrials;


if agent.test == 1
    retrain.state = 2*ones(nTrials);
    R = [1 0; 0 1];  % change reward
    
    phi = [0;1;1];
    % 50 trials retrain
    for t = 1:length(retrain.state)
        s = retrain.state(t);
        
        % policy
        d = agent.beta*(theta'*phi)' + log(p);
        logpolicy = d - logsumexp(d);
        policy = exp(logpolicy);    % softmax policy
        a = fastrandsample(policy); % action
        
        %r = rand < R(s,a);           % reward
        r = R(s,a);
        
        cost = logpolicy(a) - log(p(a));                        % policy complexity cost
        
        % learning updates
        rpe = agent.beta*r - cost - V(s);                       % reward prediction error
        g = agent.beta*phi*(1 - policy(a));                         % policy gradient
        theta(:,a) = theta(:,a) + (agent.lrate_theta)*rpe*g;    % policy parameter update
        
        V = V + agent.lrate_V*rpe*phi;
        p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
        simdata.retrain.action(t) = a;
        simdata.retrain.reward(t) = r;
        simdata.retrain.state(t) = s;
        
    end
    simdata.retrain.pa = p;
    theta
    p
    phi = [1 0 1;
        0 1 1]';
    for i = 1:2
        d = agent.beta*(theta'*phi(:,i))' + log(p);
        logpolicy = d - logsumexp(d,2);
        policy = exp(logpolicy);    % softmax
        simdata.retrain.pas(i,:) = policy;
    end
    simdata.retrain.KL = nansum(simdata.retrain.pas.*log(simdata.retrain.pas./simdata.retrain.pa),2); % KL div for each state (row), before and after reversal (column)
    simdata.retrain.V = sum(simdata.retrain.reward==1)/length(retrain.state);
    
    
    test.state = ones(1,50);
    phi = [1;0;1];
    % 50 trials test
    for t = 1:length(test.state)
        
        s = test.state(t);
        
        % policy
        d = agent.beta*(theta'*phi)' + log(p);
        logpolicy = d - logsumexp(d);
        policy = exp(logpolicy);    % softmax policy
        a = fastrandsample(policy); % action
        
        %r = rand < R(s,a);           % reward
        r = R(s,a);
        
        cost = logpolicy(a) - log(p(a));                        % policy complexity cost
        
        % learning updates (freeze)
        %rpe = agent.beta*r - cost - V(s);                       % reward prediction error
        %g = agent.beta*phi*(1 - policy(a));                         % policy gradient
        %theta(:,a) = theta(:,a) + (agent.lrate_theta)*rpe*g;    % policy parameter update
        
        %V = V + agent.lrate_V*rpe*phi;
        
        %p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
        simdata.test.action(t) = a;
        simdata.test.reward(t) = r;
        simdata.test.state(t) = s;
        
    end
    simdata.test.pa = p;
    simdata.test.theta = theta;
    
    phi = [1 0 1;
        0 1 1]';
    for i = 1:2
        d = agent.beta*(theta'*phi(:,i))' + log(p);
        logpolicy = d - logsumexp(d,2);
        policy = exp(logpolicy);    % softmax
        simdata.test.pas(i,:) = policy;
    end
    simdata.test.KL = nansum(simdata.test.pas.*log(simdata.test.pas./simdata.test.pa),2); % KL div for each state (row), before and after reversal (column)
    simdata.test.V = sum(simdata.test.reward ==1)/nTrials;
    
end
