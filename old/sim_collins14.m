function [simdata, simresults] = sim_collins14(data,results)
    
    % Simulate Collins et al. (2014) experiment using fitted actor-critic agent.
    
    rng(1); % set random seed for reproducibility
    
    if nargin < 1
        data = load_data;
    end
    
    if nargin < 2
        load model_fits;
    end
    
    for s = 1:length(data)
        x = results(1).x(s,:);
        agent.beta = x(1);
        agent.lrate_theta = x(2);
        agent.lrate_V = x(3);
        agent.lrate_p = x(4);
        
        simdata(s) = actor_critic(agent,data(s));
    end
    
    simresults = analyze_collins14(simdata);
    
    plot_figures('fig2',simresults,simdata);
    plot_figures('fig3',simresults,simdata);