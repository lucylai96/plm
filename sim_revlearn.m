function simdata = sim_revlearn(agent)
%% reversal learning
rng(0)

simdata.tpb = 16;
simdata.nRevs = 10; % number of reversals

% reward probabilities
R1 = [1 0];
R2 = [0 1];
%R1 = [1 -1];
%R2 = [-1 1];
%R1 = [0.8 0.2];
%R2 = [0.2 0.8];


state = [];
for i = 1:simdata.nRevs
    state = [state ones(1,simdata.tpb)+mod(i,2)];
end

state = flipud(state')';

nS = 2;   % # states
nA = 2;   % # actions
theta = zeros(nS,nA);           % policy parameters (8 states, 2 actions)
V = zeros(nS,1);                % state value weights
p = ones(1,nA)/2;               % marginal action probabilities
s = 1; % what state the agent thinks they're in
a = 1;
for t = 1:length(state)          % trials
    st = state(t);
    d = agent.beta*theta(s,:) + log(p);
    logpolicy = d - logsumexp(d);
    policy = exp(logpolicy);    % softmax policsy
    a = fastrandsample(policy); % action
    
    simdata.state(t) = s;
    
    if st == 1
        %r = rand < R1(a);   % reward
        r = R1(a);           % reward
        simdata.corchoice(t) = a == 1;
    else
        %r = rand < R2(a);   % reward
        r = R2(a);           % reward
        simdata.corchoice(t) = a == 2;
    end
    
    if r == 0 % context change
        if s == 1 % reverse belief
            s = 2;
        elseif s == 2
            s = 1;
        end
    end
    
    cost = logpolicy(a) - log(p(a));                              % policy complexity cost
    
    rpe = agent.beta*r - V(s) - cost;                             % reward prediction error
    g = agent.beta*(1 - policy(a));                               % policy gradient
    theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;          % policy parameter update
    V(s) = V(s) + agent.lrate_V*rpe;
    
    p = p + agent.lrate_p*(policy - p); p = p./nansum(p);         % marginal update
    
    simdata.action(t) = a;
    simdata.reward(t) = r;
end

simdata.trueS = state;
simdata.belief = sum(simdata.state==simdata.trueS)/length(state);

% valid lose-stay: not rewarded for choosing the poorer stimulus and subsequently stayed with the same stimulus;
% find on trial t, (r = 0 && a ~= st) && on trial t+1 a ~=st
simdata.losestay = sum((simdata.reward==0 & simdata.action~=state) & ([simdata.action(2:end)~=state(2:end) 0]))/length(state);

% valid win-shift: rewarded for choosing the better stimulus and subsequently shifted to the alternate stimulus.
% find on trial t, (r = 1 && a == st) && on trial t+1: a ~=st
simdata.winshift = sum((simdata.reward==1 & simdata.action==state) & ([simdata.action(2:end)~=state(2:end) 0]))/length(state);

% invalid Lose-shift: not rewarded for choosing the better stimulus and subsequently shifted to the alternate stimulus
% find on trial t, (r = 0 && a == st) && on trial t+1: a ~=st
%simdata.invloseshift = sum((simdata.reward==0 & simdata.action==state) & ([simdata.action(2:end)~=state(2:end) 0]))/length(state);

%[simdata.state;simdata.action;simdata.reward]
end