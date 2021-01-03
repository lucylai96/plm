function [simdata, agent, simresults] = sim_collins14_old(data)
    
    % Simulate Collins et al. (2014) experiment using actor-critic agent.
    
    rng(1); % set random seed for reproducibility
    
    if nargin < 1
        data = load_data;
    end
    
    lrate_theta = [0.6 0.2];            % learning rate for policy parameters
    
    for s = 1:length(data)
        agent(s).lrate_V = 0.6;         % learning rate for state value
        agent(s).lrate_p = 0.6;         % learning rate for marginal action probabilities
        agent(s).beta = unifrnd(0,30);  % cost coefficient
        agent(s).invtemp = 5;           % inverse temperature
        agent(s).lrate_theta = lrate_theta(data(s).cond+1);
        simdata(s) = actor_critic(agent(s),data(s));
    end
    
    simresults = analyze_collins14(simdata);