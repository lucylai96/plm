function results = analyze_collins14(data)
    
    % Construct reward-complexity curve for Collins et al. (2014) data.
    
    if nargin < 1
        data = load_data;
    end
    
    beta = linspace(0.1,10,50);
    beta = logspace(log10(0.1),log10(10),50);
    
    % run Blahut-Arimoto
    for s = 1:length(data)
        B = unique(data(s).learningblock);
        setsize = zeros(length(B),1);
        R_data =zeros(length(B),1);
        V_data =zeros(length(B),1);
        for b = 1:length(B)
            ix = data(s).learningblock==B(b);
            state = data(s).state(ix);
            c = data(s).corchoice(ix);
            action = data(s).action(ix);
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
            end
            
            [R(b,:),V(b,:)] = blahut_arimoto(Ps,Q,beta);
            
            setsize(b) = length(S)-1;
            
        end
        
        for c = 1:max(setsize)
            results.R(s,:,c) = nanmean(R(setsize==c,:));
            results.V(s,:,c) = nanmean(V(setsize==c,:));
            results.R_data(s,c) = nanmean(R_data(setsize==c));
            results.V_data(s,c) = nanmean(V_data(setsize==c));
        end
        
        clear R V
        
    end
    
    % compute bias
    R = squeeze(nanmean(results.R));
    V = squeeze(nanmean(results.V));
    for c = 1:max(setsize)
        Vd2(:,c) =  interp1(R(:,c),V(:,c),results.R_data(:,c));
        results.bias(:,c) = Vd2(:,c) - results.V_data(:,c);
        results.V_interp(:,c) = Vd2(:,c);
    end
    
    
    % prettystuff
    figure; hold on;
    subplot 221; hold on;
    p = plot(mean(R'),mean(V'),'LineWidth',4);
    n = size(beta,2);
    cd = [uint8(brewermap(n,'Blues')*255) uint8(ones(n,1))].';
    drawnow
    set(p.Edge, 'ColorBinding','interpolated', 'ColorData',cd)

    xlabel('Policy complexity')
    ylabel('Average reward')
    prettyplot
    axis square
    subplot 222;
    p = plot(beta,mean(R'),'-','LineWidth',4)
     drawnow
    set(p.Edge, 'ColorBinding','interpolated', 'ColorData',cd)
    xlabel('\beta')
    ylabel('Policy complexity')
    prettyplot
    axis square
    subplot 223;
    p = plot(beta,mean(V'),'-','LineWidth',4)
     drawnow
    set(p.Edge, 'ColorBinding','interpolated', 'ColorData',cd)
    xlabel('\beta')
    ylabel('Average reward')
    prettyplot
    axis square

    
    % fit empirical reward-complexity curves with polynomial
    cond = [data.cond];
    for j = 1:2
        results.bic = zeros(max(setsize),2);
        results.aic = zeros(max(setsize),2);
        for c = 1:max(setsize)
            x = results.R_data(cond==j-1,c);
            x = [ones(size(x)) x x.^2];
            y = results.V_data(cond==j-1,c);
            n = length(y); k = size(x,2);
            [b,bint] = regress(y,x);
            results.bci_sep(c,j,:) = diff(bint,[],2)/2;
            results.b_sep(c,j,:) = b;
            mse = mean((y-x*b).^2);
            results.bic(c,1) = results.bic(c,1) + n*log(mse) + k*log(n);
            results.aic(c,1) = results.bic(c,1) + n*log(mse) + k*2;
            
            x = results.R_data(:,c);
            x = [ones(size(x)) x x.^2];
            y = results.V_data(:,c);
            n = length(y); k = size(x,2);
            [b,bint] = regress(y,x);
            results.bci_joint(c,j,:) = diff(bint,[],2)/2;
            results.b_joint(c,j,:) = b;
            mse = mean((y-x*b).^2);
            results.bic(c,2) = n*log(mse) + k*log(n);
            results.aic(c,2) = n*log(mse) + k*2;
            
        end
    end