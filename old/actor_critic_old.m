function simdata = actor_critic_old(agent,data)
    
    % Simulate actor-critic agent.
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
            logpolicy = agent.invtemp*theta(s,:) - logsumexp(agent.invtemp*theta(s,:));
            policy = exp(logpolicy);    % softmax policy
            a = fastrandsample(policy); % action
            if a == corchoice(t)
                r = 1;                  % reward
            else
                r = 0;
            end
            cost = logpolicy(a) - log(p(a));    % policy complexity cost
            rpe = r - cost/agent.beta - V(s);   % reward prediction error
            theta(s,a) = theta(s,a) + agent.lrate_theta*rpe*agent.invtemp*(1 - policy(a))/t;    % policy parameter update
            V(s) = V(s) + agent.lrate_V*rpe;                                                    % state value update
            p = p + agent.lrate_p*(policy - p); p = p./sum(p);                                  % marginal update
            simdata.action(ix(t)) = a;
            simdata.reward(ix(t)) = r;
        end
    end