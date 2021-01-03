function results = analyze_collins2(s,B)

% Analyze Collins (2018) data for example subject
% input: s - example subject #
%        B - block #(s)


data = load_data('collins18');

beta = linspace(0.1,15,50);

% for set size = 3
% < 0.2 complexity 13 (block 4 and block 10)
% > 0.5 complexity 29(block 14), 31 (block 1, 10, 12), 51

% for set size = 6
% < 0.2 complexity 13, 26
% > 0.8 complexity 60, 77

%B = unique(data(s).learningblock);
cond = zeros(length(B),1);
R_data =zeros(length(B),1);
V_data =zeros(length(B),1);
for b = 1:length(B)
    ix = data(s).learningblock==B(b) & data(s).phase==0;
    state = data(s).state(ix);
    c = data(s).corchoice(ix);
    action = data(s).action(ix);
    action(action==-1) = 2;
    for a = 1:max(action)
        Pa(a) = mean(action==a);            % marginal action probability
    end
    R_data(b) = mutual_information(state,action,0.1);
    V_data(b) = mean(data(s).reward(ix));
    
    S = unique(state);
    Q = zeros(length(S),3);
    Ps = zeros(1,length(S));
    for i = 1:length(S)
        ii = state==S(i);
        Ps(i) = mean(ii);
        a = c(ii); a = a(1);
        Q(i,a) = 1;
        for a = 1:max(action)
            Pas(i,a) = mean(action(state==i)==a); % conditioned on state
        end
    end
    
    KL = nansum(Pas.*log(Pas./Pa),2); % KL divergence between policies for each state
    %results.complex(b) = sum(Ps'.*KL);
    
    if length(S)==3
        cond(b) = 1;
    else
        cond(b) = 2;
    end
    
    results.KL(:,:,b) = KL;
    results.Pa(b,:) = Pa;
    results.Pas(:,:,b) = Pas;
    clear Pa Pas KL
end


for c = 1:2
    results.R_data(s,c) = nanmean(R_data(cond==c));
    results.V_data(s,c) = nanmean(V_data(cond==c));
    
end
end
