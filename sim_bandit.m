function simdata = sim_bandit(agent)
% contextual bandits

rng(1)

nS = 2;   % # state features
nA = 2;   % # actions
theta = zeros(nS,nA);                 % policy parameters (13 state-features, 4 actions)
V = zeros(nS,1);                      % state value weights
p = ones(1,nA)/nA;                    % marginal action probabilities

% A B, L R
%R = [0.8 0.8; 0.2 0.2];
R = [1 0; 1 0];
state = 1:2;                     % discretized indices of stimuli
state = repmat(state, 1, 50);   % stim repeats, there will be rep x 2 trials
%state = state(randperm(length(state)));

% 100 trials
for t = 1:length(state)
    s = state(t);
    
    % policy
    d = agent.beta*theta(s,:) + log(p);
    logpolicy = d - logsumexp(d);
    policy = exp(logpolicy);    % softmax policy
    a = fastrandsample(policy); % action
    
    if t == length(state)/2
        %R = [0.8 0.2; 0.2 0.8];  % change reward
        R = [1 0; 0 1];  % change reward
        p
        theta
        
        simdata.pas(:,:,1) = exp([agent.beta*(theta) + log(p)] - logsumexp([agent.beta*(theta) + log(p)],2));
    end
    
    %r = rand < R(s,a);           % reward
    r = R(s,a);
    
    cost = logpolicy(a) - log(p(a));                        % policy complexity cost
    
    % learning updates
    rpe = agent.beta*r - cost - V(s);                       % reward prediction error
    g = agent.beta*(1 - policy(a));                         % policy gradient
    theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;    % policy parameter update
    
    V(s) = V(s) + agent.lrate_V*rpe;
    
    p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
    simdata.action(t) = a;
    simdata.reward(t) = r;
    simdata.state(t) = s;
    simdata.pa(t,:) = p;
    
end
p
theta
simdata.theta = theta;

d = agent.beta*(theta) + log(p);
logpolicy = d - logsumexp(d,2);
policy = exp(logpolicy);    % softmax
simdata.pas(:,:,2) = policy;

pa(:,:,1) = simdata.pa(length(state)/2,:);
pa(:,:,2) = simdata.pa(end,:);
simdata.KL = nansum(simdata.pas.*log(simdata.pas./pa),2); % KL div for each state
end
