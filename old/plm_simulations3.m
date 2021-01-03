function plm_simulations3(fig)
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
        
end
end

function data = go_nogo(S,R,cmax,beta)

%% init
% initialize variables
nS = numel(unique(S)); nA = 2;        % #states x #actions
theta = zeros(1,nA);     % policy weights
V = zeros(nS,1);         % expected value
Q = zeros(nS,nA);        % expected value
w = zeros(nS,1);         % value weights
phi = [0 0 ; 0 0 ];
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
    phi(:,1) = [0; Q(s,1)-Q(s,2)];
    phi(:,2) = [V(s); Q(s,2)-Q(s,1)];
    %act = beta*[theta'*phi]+[log(pa(2))-log(pa(1))];  % p(Go)
    act = [theta(:,1)'*phi(:,1) theta(:,2)'*phi(:,2)];  % p(Go)
    %act = theta'*phi;  % p(Go)
    
    %act = beta*[theta([s s+3],1)'*phi([s s+3],1) theta([s s+3],2)'*phi([s s+3],2)]+log(pa);  % action probabilities
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
    rpe = r - rho + (1/beta)*cost + gamma*(theta(:,a)'*phi(:,a))-(theta(:,a)'*phi0(:,a));
    rho = rho + alpha_r*(r-rho);
    
    % state/state-action value update
    Q(s,a) = Q(s,a)+alpha_Q*(r-Q(s,a));
    V(s) = V(s)+alpha_V*(r-V(s));
    
    % policy update
    %theta([s s+3],a) = theta([s s+3],a) + alpha_t*rpe*phi0([S(t-1) S(t-1)+3],a);         % policy weight update
    theta(:,a) = theta(:,a) + alpha_t*rpe*phi0(:,a);         % policy weight update
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
data.pi = exp(theta);
data.pi = data.pi./sum(data.pi,2);


data.hitg(isnan(data.hitg)) = [];
data.hitng(isnan(data.hitng)) = [];
if length(data.hitg)<length(data.hitng)
    data.hitg = [data.hitg 0];
elseif length(data.hitg)>length(data.hitng)
    data.hitng = [data.hitng 0];
end
data.gobias = movmean(movmean(data.hitg,5)-movmean(data.hitng,5),5);
Q
V
phi

%data.pi = beta*[theta'*[V Q(:,2)-Q(:,1)]'] + [log(pa(2))-log(pa(1))]
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
theta
mean(data.gobias)

%rng('default')       % for reproducibility
X = rand(3,2);
%squareform(pdist(X)) % euclidean
end



function data = go_nogo2(S,R,cmax,beta)

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
    phi(:,1) = [0; Q(s,1)-Q(s,2)];
    phi(:,2) = [V(s); Q(s,2)-Q(s,1)];
    %act = beta*[theta'*phi]+[log(pa(2))-log(pa(1))];  % p(Go)
    act = [theta(:,1)'*phi(:,1) theta(:,2)'*phi(:,2)];  % p(Go)
    %act = theta'*phi;  % p(Go)
    
    %act = beta*[theta([s s+3],1)'*phi([s s+3],1) theta([s s+3],2)'*phi([s s+3],2)]+log(pa);  % action probabilities
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
    rpe = r - rho + (1/beta)*cost + gamma*(theta(:,a)'*phi(:,a))-(theta(:,a)'*phi0(:,a));
    rho = rho + alpha_r*(r-rho);
    
    % state/state-action value update
    Q(s,a) = Q(s,a)+alpha_Q*(r-Q(s,a));
    V(s) = V(s)+alpha_V*(r-V(s));
    
    % policy update
    %theta([s s+3],a) = theta([s s+3],a) + alpha_t*rpe*phi0([S(t-1) S(t-1)+3],a);         % policy weight update
    theta(:,a) = theta(:,a) + alpha_t*rpe*phi0(:,a);         % policy weight update
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
data.pi = exp(theta);
data.pi = data.pi./sum(data.pi,2);


data.hitg(isnan(data.hitg)) = [];
data.hitng(isnan(data.hitng)) = [];
if length(data.hitg)<length(data.hitng)
    data.hitg = [data.hitg 0];
elseif length(data.hitg)>length(data.hitng)
    data.hitng = [data.hitng 0];
end
data.gobias = movmean(movmean(data.hitg,5)-movmean(data.hitng,5),5);

%data.pi = beta*[theta'*[V Q(:,2)-Q(:,1)]'] + [log(pa(2))-log(pa(1))]
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
theta
mean(data.gobias)

%rng('default')       % for reproducibility
X = rand(3,2);
%squareform(pdist(X)) % euclidean
end