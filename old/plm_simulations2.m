function plm_simulations2(fig)
% PURPOSE: policy search applied to diff experiments
% organize by the results we're trying to simulate
% INPUTS:
%
% OUTPUTS:
%
%
% Written by: Lucy Lai

addpath('/Users/lucy/Google Drive/Harvard/Projects/rc-behav/code/GoNogo-control/')
addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools')

map = brewermap(4,'*RdBu');
temp = map(3,:);
map(3,:) = map(4,:);
map(4,:) = temp;  % swap colors so darker ones are fixed
set(0, 'DefaultAxesColorOrder', map) % first three rows
set(0, 'DefaultLineLineWidth', 2) % first three rows





switch fig
    
    %% general case
    case 0
        % define features
        phi0 = phi;
        phi0(:,1) = []; % a = 1
        phi0(:,2) = []; % a = 2
        
        % action selection
        act = beta*[theta(:,1)'*phi(:,1) theta(:,2)'*phi(:,2)]+log(pa);  % action probabilities
        value = [theta(:,1)'*phi(:,1) theta(:,2)'*phi(:,2)];
        habit = log(pa);
        pi_as = [1./(1+exp(-(act(1)-act(2)))) 1-(1./(1+exp(-(act(1)-act(2)))))];
        
        % policy sampled stochastically
        if rand < pi_as(1)
            a(t) = 1;      % null
        else
            a(t) = 2;      % press lever
        end
        
        % action history
        pa = pa + alpha_pi*(pi_as - pa);
        
        % policy complexity
        cost0 = log(pi_as./pa);
        cost = cost0(a(t));
        
        % sample reward
        % reward
        r = ~test(t)*double(x(t)==2)-acost*(a(t)==2);
        rpe = r + (gamma/beta)*(log(pa(a(t)))*exp(beta*(w'*phi(:,a(t)))))-(w'*phi0(:,a(t)));
        if ecost > agent.cmax && k == 1% if I> C
            keyboard
            k = 0;
            
        else                  % increase beta otherwise
            beta = beta + alpha_b;
        end
        
        % avg reward update
        rho0 = rho;
        rho =  rho + alpha_r*(r-rho);
        
        % value update
        w0 = w;
        w = w + alpha_w*rpe*phi0(:,a(t));                                 % weight update with value gradient
        
        % policy update
        theta0 = theta;
        theta(:,a(t)) = theta(:,a(t)) + alpha_t*rpe*phi0(:,a(t));         % policy weight update
        
        %% Dorfman and Gershman
    case 1
        % feature definition:
        % the features are the tabular state-action values and the
        % state-values
        
        % compression arises in the LC case because load
        % habit is a form of compression (tradeoff between complexity and
        % within subjects-- LC and HC should achieve diff rates of reward
        % for the same policy complexity
        
        % pick different
        
        % reward), each point achieves a diff tradeoff
        %
        %
        % cases where compression arises in LC/HC cases
        %
        % inputs:
        %   x - [T x 1] vector containing the sequence of stimuli (cues) over time T
        %   R - [S x A] matrix of stimulus-action reward probabilities
        %               (e.g. R = [0.5 0.5] for 'uncontrollable' environment and
        %               R = [0.9 0.1] for 'controllable' environment if S = 1 and A:{Go, No-Go}
        cc = [0.1:0.1:0.5];
        data = load_data('data2.csv');
        %load results2
        for c = 1:length(cc)
            for i = 1:10 %:length(data) % for each subject
                s = 1;
                B = unique(data(s).block);
                for b = 1:length(B)           % for each block type (2 blocks: LC, HC)
                    ix = data(s).block==B(b); % get trials only from that block (1: LC, 2: HC)
                    S = data(s).s(ix);    % state is 1:6, 1-3 is LC: {go-to-win, nogo-to-win, decoy}; 4-6 is HC: {go-to-win, nogo-to-win, decoy}
                    A = data(s).a(ix);
                    beta = 0.1; cmax = cc(c);
                    if max(unique(S))==3 %LC
                        R = [0.25 0.75; 0.75 0.25; 0.5 0.5];
                        data(s).lc = go_nogo(S,R,cmax, beta);  % low control
                    else % HC
                        R = [0.25 0.75; 0.75 0.25; 0.2 0.8]; % cols =(ng, g)
                        S = S-3;
                        data(s).hc = go_nogo(S,R,cmax, beta);  % high control
                    end
                    
                end
                
                figure; hold on;
                subplot 221; hold on;
                %plot(data(s).lc.pa(:,2));
                plot(data(s).lc.gobias,'r'); ylabel('Go bias');
                subplot 222; hold on;
                plot(data(s).lc.cost,'r'); plot(data(s).lc.ecost,'k'); ylabel('cost');legend('C(\pi)','E[C(\pi)]')
                subplot 223; hold on;
                plot(data(s).lc.beta,'r'); ylabel('\beta');
                subplot 224; hold on;
                scatter(data(s).lc.ecost,data(s).lc.rho, 20, brewermap(length(data(s).lc.ecost),'Reds'), 'filled'); ylabel('\rho'); xlabel('E[C(\pi)]');
                scatter(data(s).lc.ecost(end),data(s).lc.rho(end), 50, 'k', 'filled');
                subprettyplot(2,2)
                suptitle('LC')
                
                
                figure; hold on;
                subplot 221; hold on;
                %plot(data(s).hc.pa(:,2));plot(data(s).hc.gobias);  ylabel('p(a=Go)')
                plot(data(s).hc.gobias); ylabel('Go bias');
                subplot 222; hold on;
                plot(data(s).hc.cost); plot(data(s).hc.ecost); ylabel('cost');legend('C(\pi)','E[C(\pi)]')
                subplot 223; hold on;
                plot(data(s).hc.beta); ylabel('\beta');
                subplot 224; hold on;
                scatter(data(s).hc.ecost,data(s).hc.rho, 20, brewermap(length(data(s).hc.ecost),'Blues'), 'filled'); ylabel('\rho'); xlabel('E[C(\pi)]');
                scatter(data(s).hc.ecost(end),data(s).hc.rho(end), 50, 'k', 'filled');
                subprettyplot(2,2)
                suptitle('HC')
                
                figure; bar(mean([data(s).lc.gobias; data(s).hc.gobias],2));
                set(gca,'xtick',[1:2],'xticklabel',{'LC' 'HC'})
                ylabel('Go bias')
                xlabel('Condition')
                prettyplot
                
                
                figure(100+c); hold on; plot([data(s).lc.pi(3,2)],mean(data(s).lc.gobias),'r.','MarkerSize', 40)
                plot([data(s).hc.pi(3,2)],mean(data(s).hc.gobias),'b.','MarkerSize', 40)
                ylabel('Go bias');xlabel('\pi(a=Go|s=decoy)')
                legend('LC','HC')
                prettyplot
                
                GO(c,i,1) = mean(data(s).lc.gobias);
                GO(c,i,2) = mean(data(s).hc.gobias);
                THETA(c,i,1) = data(s).lc.pi(3,2);
                THETA(c,i,2) = data(s).hc.pi(3,2);
            end
            close all
        end
          figure; subplot 121;hold on; 
        k1 = shadedErrorBar(cc,mean(GO(:,:,1),2), sem(GO(:,:,1),2),'r',1);
        k2 = shadedErrorBar(cc,mean(GO(:,:,2),2), sem(GO(:,:,2),2),'b',1);
        ylabel('Go bias');xlabel('C_{max}')
        prettyplot
        
        subplot 122; hold on;
        shadedErrorBar(cc,mean(THETA(:,:,1),2), sem(THETA(:,:,1),2),'r',1);
        shadedErrorBar(cc,mean(THETA(:,:,2),2), sem(THETA(:,:,2),2),'b',1);
        ylabel('\pi(a=Go|s=decoy)');xlabel('C_{max}')
        prettyplot
        why
        %%
        
    case 2
        
        
    case 3
        
        
        
        
end % end switch

%% plot
figure; hold on;

subplot 221; hold on;
plot(data.s2.theta);
subplot 222;  hold on;
plot(data.s2.gobias,'Color',map(1,:));

subplot 221; hold on;
plot(data.s4.theta);
legend('\theta_1, set size = 2','\theta_2, set size = 2','\theta_1, set size = 4','\theta_2, set size = 4'); legend('boxoff');
xlabel('trial')
ylabel('parameter value');

subplot 222;  hold on;
plot(data.s4.gobias,'Color',map(3,:));
xlabel('trial')
ylabel('Go bias')
legend('set size = 2','set size = 4'); legend('boxoff');


subplot 223; hold on;
plot(data.s2.cost,'-')
plot(data.s4.cost,'-')
xlabel('timesteps')
ylabel('policy cost C(\pi_\theta)')
subplot 224; hold on;
plot(data.s2.rpe)
plot(data.s4.rpe)
xlabel('timesteps')
ylabel('rpe')
subprettyplot(2,2)

suptitle(strcat('\beta =',num2str(beta)))
end

function data = go_nogo(S,R,cmax,beta)

%% init
% initialize variables
nS = numel(unique(S)); nA = 2;        % #states x #actions
theta = zeros(1,nA);     % policy weights
V = zeros(nS,1);         % expected value
Q = zeros(nS,nA);        % expected value
w = zeros(nS,1);         % value weights
phi = [V V; Q];
theta = zeros(size(phi));

ecost = 0;
pa = [0.5 0.5];
gamma = 0.98;
rho = 0;

% learning rates
alpha_Q = 0.1;         % Q learning rate
alpha_V = 0.1;         % V learning rate
alpha_pi = 0.1;        % marginal p learning rate
alpha_b = 0.1;         % beta learning rate
alpha_t = 0.1;         % theta learning rate
alpha_e = 0.1;         % exp cost learning rate
alpha_r = 0.1;         % avg rew learning rate
Ps = [0 0 0];
Pa = [0 0];
data.hitg = NaN(1,length(S)); data.hitng = NaN(1,length(S));
%% loop through stimuli
k = 1;
for t = 2:length(S)
    s = S(t);
    Ps(s) = Ps(s)+1;
    
    if s>3
        keyboard
    end
    % action selection
    %act = beta*Q(s,:)+log(pa);  % action probabilities
    phi0 = phi;
    phi(:,1) = [0; 0; 0; Q(:,1)];
    phi(:,2) = [V; Q(:,2)];
   
    %act = beta*[theta([s s+3],1)'*phi([s s+3],1) theta([s s+3],2)'*phi([s s+3],2)]+log(pa);  % action probabilities
    act = [theta([s s+3],1)'*phi([s s+3],1) theta([s s+3],2)'*phi([s s+3],2)];  % action probabilities
    %value = Q(s,:);
    %habit = log(pa);
    %pi_as = [1-(1./(1+exp(-act))) 1./(1+exp(-act))];
    
    %value = [theta(:,1)'*phi(:,1) theta(:,2)'*phi(:,2)];
    %value = [theta(2:end,1)'*phi(2:end,1) theta(2:end,2)'*phi(2:end,2)];
    %habit = log(pa);
    %habit = [theta(1,1)'*phi(1,1) theta(1,2)'*phi(1,2)];
    pi_as = [1./(1+exp(-(act(1)-act(2)))) 1-(1./(1+exp(-(act(1)-act(2)))))];
    
    if rand < pi_as(1)
        a = 1;   % no-go
    else
        a = 2;   % go
    end
    
    % sample reward
    r = rand < R(s,a);
    
    % action history
    pa = pa + alpha_pi*(pi_as - pa);
    Pa(a) = Pa(a) + 1;
    
    % policy complexity
    cost0 = log(pi_as./pa);
    cost = cost0(a);
    ecost = ecost+alpha_e*(cost-ecost);
    
    % reward
    %rpe = r + (gamma/beta)*(log(pa(a))*exp(beta*Q(s,a)))-Q(S(t-1),a);
    %rpe = r + (gamma/beta)*(log(pa(a))*exp(beta*(theta([s s+3],a)'*phi([s s+3],a))))-(theta([S(t-1) S(t-1)+3],a)'*phi0([S(t-1) S(t-1)+3],a));
    rpe = r - rho + (1/beta)*cost + gamma*(theta([s s+3],a)'*phi([s s+3],a))-(theta([S(t-1) S(t-1)+3],a)'*phi0([S(t-1) S(t-1)+3],a));
    rho = rho + alpha_r*(r-rho);
    
    % state/state-action value update
    Q(s,a) = Q(s,a)+alpha_Q*(r-Q(s,a));
    V(s) = V(s)+alpha_V*(r-V(s));
    
    % policy update
    %theta([s s+3],a) = theta([s s+3],a) + alpha_t*rpe*phi0([S(t-1) S(t-1)+3],a);         % policy weight update
    theta([s s+3],a) = theta([s s+3],a) + alpha_t*rpe*phi0([S(t-1) S(t-1)+3],a);         % policy weight update
    % why would theta for V(s) be any different than theta for Q(s,a) if
    % both are subject to same cost penality? the thing that would reduce
    % cost (compress) is the repetition of action and not state-sensitive
    % information
    
    
    %PA = Pa/sum(Pa); % normalize PA
    %PA = PA + 0.1; PA = PA./sum(PA);
    %mi = sum(pi_as.*log(pi_as./PA)); % for this state at this timestep
    
    % beta update
    if (ecost > cmax) && k == 1% if I> C
        disp('!')
        %beta = beta - alpha_b; % increase beta otherwise
        %keyboard
        k = 0;
    elseif k == 1
        %beta = beta + alpha_b; % increase beta otherwise
    end
    
    
    % go-bias
    if s == 1
        if a == 2 % go to win -- hit
            data.hitg(t) = 1;
        else
            data.hitg(t) = 0;
        end
    elseif s == 2
        if a == 1 % no-go to win  -- hit
            data.hitng(t) = 1;
        else
            data.hitng(t) = 0;
        end
    end
    
    
    % store results
    data.a(t) = a;
    data.r(t) = r;
    data.pi_as(t,:) = pi_as;
    data.s(t) = s;
    data.pa(t,:) = pa;
    %data.mi(t) = mi;
    data.rpe(t) = rpe;
    data.rho(t) = rho;
    %data.phi(:,:,t) = phi;
    data.Q(:,:,t) = Q;
    data.cost(t) = cost;
    data.ecost(t) = ecost;
    data.beta(t) = beta;
    
    
end
%data.pi = exp(beta*[theta.*phi]+log(pa));  % action probabilities
for s=1:3
    data.pi(s,:) = exp([theta([s s+3],1)'*phi([s s+3],1) theta([s s+3],2)'*phi([s s+3],2)]);
end
data.pi = data.pi./sum(data.pi,2);


data.hitg(isnan(data.hitg)) = [];
data.hitng(isnan(data.hitng)) = [];
if length(data.hitg)<length(data.hitng)
    data.hitg = [data.hitg 0];
elseif length(data.hitg)>length(data.hitng)
    data.hitng = [data.hitng 0];
end
data.gobias = movmean(movmean(data.hitg,5)-movmean(data.hitng,5),5);



data.theta = theta;
%figure;imagesc(data.pi); colorbar; colormap(flipud(gray));

%caxis([0 1])
%title('\pi(a|s)')
%set(gca,'xtick',[1:2],'xticklabel',{'no-go' 'go'})

%figure;imagesc(theta'); colorbar; colormap(flipud(gray));
%title('\theta')
%set(gca,'xtick',[1:2],'xticklabel',{'V(s)' 'Q(s,a)'})

%prettyplot
%figure; plot(data.mi)
%hold on; plot(data.ecost)


% plot state x state similarity
mean(data.gobias)

%rng('default')       % for reproducibility
X = rand(3,2);
%squareform(pdist(X)) % euclidean
end
