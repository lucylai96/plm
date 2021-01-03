function main_sz(hyp)
% PURPOSE: to simulate policy gradient for Collins et al. (2014) data
% INPUT: hyp is the hypothesis (as detailed below)

% NOTES:
% If you were able to simulate policy gradient on this task,
% and show that (a) learning rate determines divergence from
% the optimal trade-off curve, and (b) the schizophrenic data
% can be captured by assuming a different value of beta, but not
% necessarily a difference in learning rate, then I think I would
% be able to respond to the remaining reviewer comments (I've already dealt with a good number of them).

% Both be free parameters (1) learning rate or (2) beta to be the free parameter and see which explains the difference between HC and SZ the best?  .
% Once you have the model fits, you'd simulate data using those models and the experimental paradigm,
% so produce simulated reward-complexity functions. And hopefully this can reproduce the features that
% (a) SZ and HC are on the same empirical curve, and (b) higher deviation for lower complexity policies


data = load_data;


switch hyp
      %% hypothesis 1: same betas but different learning rates explain suboptimality and difference between HC/SCZ
    case 1
        close all
        % 'HC': different learning rates, but same betas
        beta = linspace(0.1,15,50);     % 20 values of betas from 1-20
        alpha_t_hc = linspace(0.01,0.8,5);   % 5 learning rates from 0.01 to 1
        alpha_w = ones(1,length(alpha_t_hc))*0.5;  % fixed value learning rate
        alpha_w = alpha_t_hc;  % value learning rate = policy learning rate
        
        colstr = 'b'; % plot color (HC)
        sim_loop(data, beta, alpha_t_hc, alpha_w, colstr) % loop over simulations with the specified learning rates / betas
        
        % 'SCZ': different learning rates, but same betas
        alpha_t_sz = linspace(0.01,0.4,5);   % 5 learning rates from 0.01 to 0.5
        alpha_w = alpha_t_sz;  % value learning rate = policy learning rate
        
        colstr = 'r'; % plot color (SCZ)
        sim_loop(data, beta, alpha_t_sz, alpha_w, colstr)
        
        % format grid
        for lr = 1:length(alpha_t_hc)
            figure(lr); hold on;
            sgtitle({strcat('HC \alpha_\theta:',num2str(alpha_t_hc(lr))),strcat('SCZ  \alpha_\theta: ', num2str(alpha_t_sz(lr)))})
        end
        
        
        %% hypothesis 2: same learning rate but diff betas explain suboptimality and difference between HC/SCZ
    case 2
        close all
         % 'HC': same learning rates, but different betas
        alpha_t = linspace(0.01,0.7,5);   % 5 learning rates from 0.01 to 1
        alpha_w = ones(1,length(alpha_t))*0.5;  % value learning rate
        alpha_w = alpha_t;  % value learning rate = policy learning rate
        
        beta = linspace(0.1,15,50);     % 20 values of betas from 1-10
         
        colstr = 'b'; % plot color (HC)
        sim_loop(data, beta, alpha_t, alpha_w, colstr) % loop over simulations with the specified learning rates / betas
        
        %% plot again, with different betas ('SCZ'), but same learning rates
        beta = linspace(10,20,20);     % 20 values of betas from 10-20
        colstr = 'r'; % plot color (SCZ)
        sim_loop(data, beta, alpha_t, alpha_w, colstr)
       
        % format grid
        for lr = 1:length(alpha_t)
            figure(lr); hold on;
            sgtitle(strcat('HC & SCZ \alpha_\theta:',num2str(alpha_t(lr))))
        end
       
end





end % schiz driver

function sim_loop(data, beta, alpha_t, alpha_w, colstr)
% PURPOSE: loops over simulations with the specified learning rates / betas
% OUTPUT: makes length(alpha_t) reward-complexity plots

% looking at 1 set of "experimental stimuli" for now

for s = 1 %:length(data) % looking at 1 set of "experimental stimuli" for now
    B = unique(data(s).learningblock);
    setsize = zeros(length(B),1);
    input = data(s); % subject's data / expt structure
    
    for lr = 1:length(alpha_t)
        
        R_sim = zeros(length(B),length(beta)); % simulated R-V
        V_sim = zeros(length(B),length(beta)); % simulated R-V
        R_th = zeros(length(B),length(beta)); % theory R-V
        V_th = zeros(length(B),length(beta)); % theory R-V
        
        for bl = 1:length(B) % for all 13 blocks
            ix = data(s).learningblock==B(bl);
            
            for b = 1:length(beta) % simulate using diff values of beta
                params(1) = alpha_w(lr);
                params(2) = alpha_t(lr);
                params(3) = beta(b);
                model(lr,b) = simulate_sz(params, input); % input to get stimulus order
                
                % value and rate, organized by (block, beta) (13 x length(beta))
                V_sim(bl,b) = mean(model(lr,b).r(ix));       % avg over the trials of the same block
                R_sim(bl,b) = mean(model(lr,b).cost(ix));
                
            end % for each beta
            
            setsize(bl) = max(data(s).state(ix))-1; % set size for that block (in reality is 2-6, but here set from 1-5 for plotting purposes)
            Q = zeros(setsize(bl),max(input.action));
            Ps = zeros(1,setsize(bl));
            
            for i = 1:setsize(bl) % for each state
                Ps(i) = mean(input.state==i); % get marginal state dist
                a = input.corchoice(input.state==i); a = a(1);
                Q(i,a) = 1;
            end
            
            [R_th(bl,:),V_th(bl,:)] = blahut_arimoto(Ps,Q,beta); % theoretical R-V
            
        end % for each block
        
        figure(lr); hold on; % one figure for each learning rate
        
        % averaging over blocks with the same set size
        for c = 1:max(setsize) % iterate over all set sizes
            results.R_sim(lr,:,c) = nanmean(R_sim(setsize==c,:)); % 85 subjects x 50 betas x 5 set sizes
            results.V_sim(lr,:,c) = nanmean(V_sim(setsize==c,:));
            
            results.R_th(lr,:,c) = nanmean(R_th(setsize==c,:)); % 85 subjects x 50 betas x 5 set sizes
            results.V_th(lr,:,c) = nanmean(V_th(setsize==c,:));
            
            subplot(2,3,c); hold on; %
            plot(results.R_th(lr,:,c),results.V_th(lr,:,c),'k','LineWidth',3); % theory
            scatter(results.R_sim(lr,:,c),results.V_sim(lr,:,c),50, colstr, 'filled'); % simulation
            axis([0 1 0 1])
            ylabel('Average reward')
            xlabel('Policy complexity')
            title(strcat('Set size:',num2str(c+1)))
        end
        
        % averaging over beta values
        subplot 236; hold on;
        errorbar(unique(setsize)+1,mean(squeeze(results.R_sim(lr,:,:))), sem(squeeze(results.R_sim(lr,:,:)),1),'Color',colstr,'LineWidth',2)
        ylabel('Policy complexity'); xlabel('Set size')
        xlim([0 7])
        subprettyplot(2,3)
        
        clear R V
        
       
    end % for each policy learning rate
end % for each subject

end % sim loop

function data = simulate_sz(params, input)
% INPUT: params - model parameters
%        input - exp stimuli
%
% use the data to fix:
% (1) stimuli
% (2) actions

%% initialize
% set learning rates
alpha_w = params(1); % value learning rate
alpha_t = params(2); % theta learning rate
if length(params)>2
    beta = params(3);    % beta
else
    beta = 10;
end
alpha_pi = 0.1;
alpha_r = 0.1;
alpha_e = 0.1;
s = input.state;
% fixed parameters
gamma = 0.98;
%num = 5; % how many trials to look back
% for b = 1:length(B) % for each block
% ix = data.learningblock==B(b);

for t = 1:length(input.trial) % number of trials total
    
    % notes: for each set size, there will be a new expected cost and rho
    if input.trial(t) == 1         % if a new block starts, reset weights, get number of states and actions
        %b = input.learningblock(t); % learning block #
        nS = input.ns(t);  % # states
        nA = 3;               % # actions
        theta = zeros(nS,nA);                % policy weights
        phi = zeros(nS,1);                   % features
        pa = ones(nA,1)./nA;                 % marginal
        w = zeros(nS,1);                     % value weights
        ecost = 0;                           % trade-off parameter (lower means less capacity, higher means more)
        rho = 0;
    end
    
    % loop through stimuli
    %phi0 = phi;
    %phi = zeros(nS,1);                       % features
    %phi(s(t)) = 1;
    
    %pi_as = exp(theta(s(t),:))./sum(exp(theta(s(t),:)));
    %params
    pi_as = exp(theta(s(t),:) - logsumexp(theta(s(t),:)));
    a = fastrandsample(pi_as);
    r = input.corchoice(t) == a;
    
    pa =  pa + alpha_pi*(pi_as-pa);               % marginal policy 
    
    cost = log(pi_as./pa);                        % if pi_as and pa are pretty similar, log(1) = 0 (low cost)
    cost = cost(a);                               % policy cost for that state-action pair
    ecost = alpha_e*(cost-ecost);
    
    rpe = r - (1./beta)*cost + w(s(t));%'*phi;                 % TD error
    rho =  rho + alpha_r*(r-rho);                      % average reward update
    
    % value update
    w(s(t))= w(s(t)) + alpha_w*rpe;                           % weight update
    
    % policy update
    theta(s(t),a) = theta(s(t),a) + alpha_t*rpe*(1-pi_as(a));%phi;   % policy gradient update
    
    % store vars
    data.a(t) = a;
    data.r(t) = r;
    data.pi(t,:) = pi_as;
    %data.w(t,:) = w;
    %data.theta(:,:,t) = theta;
    %data.rpe(t) = rpe;
    data.rho(t) = rho;
    data.cost(t) = cost;
    data.ecost(t) = ecost;
end




end
