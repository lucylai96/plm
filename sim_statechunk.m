function simdata = sim_statechunk(agent)
% state chunking

nS = 13;  % # state features
nA = 4;   % # actions
nEp = 10; % # episodes to train
gamma = 0.98;
e = 1; 
theta = zeros(nS,nA);                 % policy parameters (13 state-features, 4 actions)
w = zeros(nS,1);                      % state value weights
phi  = zeros(nS,1);                   % state features
p = ones(1,nA)/nA;                    % marginal action probabilities

T = world;
chunks = [1 2 3; 4 5 6; 7 8 9];

% a = 1, "up"
% a = 2, "down"
% a = 3, "left"
% a = 4, "right"

while e <= nEp
    s = randsample(3,1); % sample start location
    r = 0;
    t = 1;
    endEp = 0;
    while t<15 % used to be while you haven't gotten reward
        
        % features
        phi0 = phi;
        phi = zeros(nS,1);
        phi(s) =  1;
        phi(nS) = 1; % last state
        for i = 1:size(chunks,1)
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
        if new_s > 6 && endEp == 0 % goal states
            r = 1;
            endEp = 1;
        %elseif loop == 1
        %    r = 1;
        %elseif s > 3
        %    r = 1;
        else                       % not goal state or absorbing state reached
            r = 0;
        end
        
        cost = logpolicy(a) - log(p(a));    % policy complexity cost
        
        % learning updates
        V0 = w'*phi0;                                           % state value
        V = w'*phi;
        rpe = agent.beta*r - cost + gamma*(V - V0);             % reward prediction error
        g = agent.beta*phi*(1 - policy(a));                     % policy gradient
        theta(:,a) = theta(:,a) + (agent.lrate_theta)*rpe*g;    % policy parameter update
        
        w = w + agent.lrate_V*rpe*phi;
        
        p = p + agent.lrate_p*(policy - p); p = p./nansum(p);        % marginal update
        simdata.action(e,t) = a;
        simdata.reward(e,t) = r;
        simdata.state(e,t) = s;
        
        s = new_s;
        t = t+1;
    end
    
    %simdata.w(:,e) = w;
    e = e+1;
end
simdata.w = w;
simdata.theta = theta;
simdata.pa = p;

d = agent.beta*(theta) + log(p);
logpolicy = d - logsumexp(d,2);
policy = exp(logpolicy);    % softmax
simdata.pas = policy;
end

function T = world

% instantiate transition matrix
T = zeros(3,3,4);  % S x S x A (9 x 9 x 3)

T(2,1,1) = 1;
T(3,2,1) = 1;
T(5,4,1) = 1;
T(6,5,1) = 1;
%T(8,7,1) = 1;
%T(9,8,1) = 1;

T(1,2,2) = 1;
T(2,3,2) = 1;
T(4,5,2) = 1;
T(5,6,2) = 1;
%T(7,8,2) = 1;
%T(8,9,2) = 1;

%T(7,4,3) = 1;
T(4,1,3) = 1;
%T(8,5,3) = 1;
T(5,2,3) = 1;
%T(9,6,3) = 1;
T(6,3,3) = 1;

T(1,4,4) = 1;
T(4,7,4) = 1;
T(2,5,4) = 1;
T(5,8,4) = 1;
T(3,6,4) = 1;
T(6,9,4) = 1;

% absorbing states
T(7,7,1:4) = 1;
T(8,8,1:4) = 1;
T(9,9,1:4) = 1;

end