function simdata = sim_statechunk(agent)
% state chunking

rng(1)

nS = 13;  % # state features
%nS = 9;  % # state features
nA = 4;   % # actions
nEp = 30; % # episodes to train
gamma = 0.98;
e = 1;
theta = zeros(nS,nA);                 % policy parameters (13 state-features, 4 actions)
V = zeros(9,1);                       % state value weights
phi  = zeros(nS,1);                   % state features
p = ones(1,nA)/nA;                    % marginal action probabilities

% set up world
rS = [7];         % reward states
ss = [3];

%rS = [1 3 7 9];  % reward states
%ss = [2 4 5 6 8];

ss = datasample(ss,nEp); % vector of state states

T = world(rS);
R = zeros(1,9); R(rS) = 1; % rewarded states

%R = [0 0 0 1 1 1 2 2 2]; % rewarded states

chunks = [1 2 3; 4 5 6; 7 8 9];

% a = 1, "up"
% a = 2, "down"
% a = 3, "left"
% a = 4, "right"

for e = 1:nEp
    s = ss(e);  % sample start location
    t = 1;
    endEp = 0;
    while endEp == 0 % used to be while you haven't gotten reward
        
        % define features
        phi = zeros(nS,1);
        phi(s) =  1;
        
        phi(nS) = 1; % one chunk
        for i = 1:size(chunks,1) % 3 chunks
            if ismember(s,chunks(i,:))
                phi(9+i) = 1;
            end
        end
        
        % policy
        d = agent.beta*(theta'*phi)' + log(p);
        logpolicy = d - logsumexp(d);
        policy = exp(logpolicy);    % softmax
        
        new_s = [];
        while isempty(new_s)              % keep sampling action until legal action
            a = fastrandsample(policy);   % sample action
            new_s = find(T(s,:,a)==1);    % sample new state from transition matrix
        end
        
        if length(new_s)>1
            keyboard
        end
        
        % reward?
        if new_s == s %== 9% goal state
            endEp = 1;
        end
        
        %if ismember(new_s,rewStates)
        r = R(new_s);
        %else                               % not goal state or absorbing state reached
        %    r = 0;
        %end
        
        cost = logpolicy(a) - log(p(a));    % policy complexity cost
        
        % learning updates
        %V0 = w'*phi0;                                           % state value
        %V = w'*phi;
        rpe = agent.beta*r - cost + gamma*V(new_s) - V(s);       % reward prediction error
        g = agent.beta*phi*(1 - policy(a));                      % policy gradient
        %g = agent.beta*(1 - policy(a));                         % policy gradient
        %theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;    % policy parameter update
        theta(:,a) = theta(:,a) + (agent.lrate_theta)*rpe*g;     % policy parameter update
        
        %w = w + agent.lrate_V*rpe*phi;
        V(s) = V(s) + agent.lrate_V*rpe;
        
        p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
        simdata.action(e,t) = a;
        simdata.reward(e,t) = r;
        simdata.state(e,t) = s;
        
        s = new_s;
        t = t+1;
    end
    
end

simdata.V = V;
simdata.theta = theta;
simdata.pa = p;

phi = [1 0 0 0 0 0 0 0 0 1 0 0 1;
    0 1 0 0 0 0 0 0 0 1 0 0 1;
    0 0 1 0 0 0 0 0 0 1 0 0 1;
    0 0 0 1 0 0 0 0 0 0 1 0 1;
    0 0 0 0 1 0 0 0 0 0 1 0 1;
    0 0 0 0 0 1 0 0 0 0 1 0 1;
    0 0 0 0 0 0 1 0 0 0 0 1 1;
    0 0 0 0 0 0 0 1 0 0 0 1 1;
    0 0 0 0 0 0 0 0 1 0 0 1 1;]';
for i = 1:9
    d = agent.beta*(theta'*phi(:,i))' + log(p);
    logpolicy = d - logsumexp(d,2);
    policy = exp(logpolicy);    % softmax
    simdata.pas(i,:) = policy;
end

simdata.KL = nansum(simdata.pas.*log(simdata.pas./simdata.pa),2);

%% if testing on a diff structure
if agent.test == 1
    % changing reward structure
    p = ones(1,nA)/nA;                    % marginal action probabilities
    
    % set up world
    rS = [3];   % reward states
    ss = [7];   % start states
    
    ss = datasample(ss,nEp); % vector of state states
    
    T = world(rS);
    R = zeros(1,9); R(rS) = 1; % rewarded states
    
    nTestEp = 5; % number of test episodes
    for e = 1:nTestEp
        s = ss(e);  % sample start location
        t = 1;
        endEp = 0;
        while endEp == 0 % used to be while you haven't gotten reward
            
            % define features
            phi = zeros(nS,1);
            phi(s) =  1;
            
            phi(nS) = 1; % one chunk
            for i = 1:size(chunks,1) % 3 chunks
                if ismember(s,chunks(i,:))
                    phi(9+i) = 1;
                end
            end
            
            % policy
            d = agent.beta*(theta'*phi)' + log(p);
            logpolicy = d - logsumexp(d);
            policy = exp(logpolicy);    % softmax
            
            new_s = [];
            while isempty(new_s)              % keep sampling action until legal action
                a = fastrandsample(policy);   % sample action
                new_s = find(T(s,:,a)==1);    % sample new state from transition matrix
            end
            
            if length(new_s)>1
                keyboard
            end
            
            % reward?
            if new_s == s %== 9% goal state
                endEp = 1;
            end
            
            %if ismember(new_s,rewStates)
            r = R(new_s);
            %else                               % not goal state or absorbing state reached
            %    r = 0;
            %end
            
            cost = logpolicy(a) - log(p(a));    % policy complexity cost
            
            % learning updates
            %V0 = w'*phi0;                                           % state value
            %V = w'*phi;
            rpe = agent.beta*r - cost + gamma*V(new_s) - V(s);       % reward prediction error
            g = agent.beta*phi*(1 - policy(a));                      % policy gradient
            %g = agent.beta*(1 - policy(a));                         % policy gradient
            %theta(s,a) = theta(s,a) + (agent.lrate_theta)*rpe*g;    % policy parameter update
            theta(:,a) = theta(:,a) + (agent.lrate_theta)*rpe*g;     % policy parameter update
            
            %w = w + agent.lrate_V*rpe*phi;
            V(s) = V(s) + agent.lrate_V*rpe;
            
            p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
            simdata.test.action(e,t) = a;
            simdata.test.reward(e,t) = r;
            simdata.test.state(e,t) = s;
            
            s = new_s;
            t = t+1;
        end
        
    end % agent test
end
simdata.test.V = V;
simdata.test.theta = theta;
simdata.test.pa = p;

phi = [1 0 0 0 0 0 0 0 0 1 0 0 1;
    0 1 0 0 0 0 0 0 0 1 0 0 1;
    0 0 1 0 0 0 0 0 0 1 0 0 1;
    0 0 0 1 0 0 0 0 0 0 1 0 1;
    0 0 0 0 1 0 0 0 0 0 1 0 1;
    0 0 0 0 0 1 0 0 0 0 1 0 1;
    0 0 0 0 0 0 1 0 0 0 0 1 1;
    0 0 0 0 0 0 0 1 0 0 0 1 1;
    0 0 0 0 0 0 0 0 1 0 0 1 1;]';

for i = 1:9
    d = agent.beta*(theta'*phi(:,i))' + log(p);
    logpolicy = d - logsumexp(d,2);
    policy = exp(logpolicy);    % softmax
    simdata.test.pas(i,:) = policy;
end

simdata.test.KL = nansum(simdata.pas.*log(simdata.pas./simdata.pa),2);
end

function T = world(rS)
% rS are rewarded states

% instantiate transition matrix
T = zeros(3,3,4);  % S x S x A (9 x 9 x 3)

% up
T(2,1,1) = 1;
T(3,2,1) = 1;
T(5,4,1) = 1;
T(6,5,1) = 1;
T(8,7,1) = 1;
T(9,8,1) = 1;

% down
T(1,2,2) = 1;
T(2,3,2) = 1;
T(4,5,2) = 1;
T(5,6,2) = 1;
T(7,8,2) = 1;
T(8,9,2) = 1;

% left
T(7,4,3) = 1;
T(4,1,3) = 1;
T(8,5,3) = 1;
T(5,2,3) = 1;
T(9,6,3) = 1;
T(6,3,3) = 1;

% right
T(1,4,4) = 1;
T(4,7,4) = 1;
T(2,5,4) = 1;
T(5,8,4) = 1;
T(3,6,4) = 1;
T(6,9,4) = 1;

% absorbing states can't go to any other state except themselves
for s = 1:length(rS)
    T(rS(s),:,1:4) = 0;     % absorbing state can't go to any other state
    T(rS(s),rS(s),1:4) = 1; % absorbing state loops to itself
    
end


end
