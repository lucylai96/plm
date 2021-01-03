function simdata = actor_critic2(agent,data)

% Simulate actor-critic agent. This time only in a 2 action task where
% 1 action is rewarded more than another
%
% USAGE: simdata = actor_critic(agent,data)

simdata = data;
B = unique(data.learningblock);
for b = 1:length(B)
    ix = find(data.learningblock==B(b));
    state = data.state(ix);
    corchoice = data.corchoice(ix);     % correct choice on each trial
    setsize = length(unique(state));    % number of distinct stimuli
    theta = zeros(setsize,3);           % policy parameters
    V = zeros(setsize,1);               % state values
    p = ones(1,3)/3;                    % marginal action probabilities
    for t = 1:length(state)
        s = state(t);
        d = agent.beta*theta(s,:) + log(p);
        logpolicy = d - logsumexp(d);
        policy = exp(logpolicy);    % softmax policy
        a = fastrandsample(policy); % action
        if a == corchoice(t) % normal reward
            r = 1;                  % reward
        else
            r = 0;
        end
        
        if setsize == 2
            if t > length(state)/2 % reversal
                
                if a == corchoice(t) % normal reward
                    r = 0;
                else
                    r = 1;
                end
                simdata.rev(ix(t)) = t;
            end % reversal
        end
        
        cost = logpolicy(a) - log(p(a));    % policy complexity cost
        if agent.nc == 1
            rpe = r - V(s);               % reward prediction error no cost
        else
            rpe = agent.beta*r - cost - V(s);   % reward prediction error
        end
        g = (1 - policy(a))*(agent.beta + policy(a)/(setsize*p(a))); % policy gradient
        theta(s,a) = theta(s,a) + rpe*agent.lrate_theta*g/t;    % policy parameter update
        V(s) = V(s) + agent.lrate_V*rpe;                                                    % state value update
        p = p + agent.lrate_p*(policy - p); p = p./sum(p);                                  % marginal update
        
        simdata.action(ix(t)) = a;
        simdata.reward(ix(t)) = r;
        simdata.rpe(ix(t)) = rpe;
        simdata.setsize(ix(t)) = setsize;
        simdata.cost(ix(t)) = cost;
        simdata.s(ix(t)) = s;
    end
end