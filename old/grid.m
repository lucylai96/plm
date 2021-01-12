function grid

%% state chunking: gridworld

agent.lrate_V = 0.1;
agent.lrate_p = 0.0;
agent.lrate_theta = 0.1;
beta = [0.1 1 1.5 2]; % capacity constraint

agent.test = 1;
for b = 1:length(beta)
    agent.beta = beta(b);
    simdata(b) = sim_statechunk(agent);
    test(b) = simdata(b).test;
end

% cost
figure; hold on; colormap(brewermap([],'Reds'))
for i = 1:length(simdata)
    % learned policy cost for each state (KL divergence between
    % p(a|s) and p(a)
    subplot(1,length(simdata),i);
    imagesc(reshape(simdata(i).KL,3,3))
    title(strcat('\beta=',num2str(beta(i))))
    axis square
end
eqcolorbar(1,length(simdata))
suptitle('policy cost')
set(gcf, 'Position',  [500, 500, 300+100*length(simdata), 200])

% value
figure; hold on; colormap(brewermap([],'Reds'))
for i = 1:length(simdata)
    subplot(1,length(simdata),i);
    imagesc(reshape(simdata(i).V,3,3))
    title(strcat('\beta=',num2str(beta(i))))
    axis square
end
eqcolorbar(1,length(simdata))
suptitle('value')
set(gcf, 'Position',  [500, 500, 300+100*length(simdata), 200])

% policy
% a = 1, "up"
% a = 2, "down"
% a = 3, "left"
% a = 4, "right"
action = {'up','down','left','right'};
figure; hold on; colormap(brewermap([],'Reds'))
for a = 1:length(action)
    for i = 1:length(simdata)
        subplot(length(action),length(simdata),(a-1)*length(simdata)+i);
        imagesc(reshape(simdata(i).pas(1:9,a),3,3))
        if i == 1
            ylabel(strcat('p(',action{a},')'))
        end
        if a == 1
            title(strcat('\beta=',num2str(beta(i))))
        end
        axis square
    end
end
caxis([0 1])
eqcolorbar(length(action),length(simdata))
suptitle('policy')
set(gcf, 'Position',  [600, 50, 300+100*length(simdata), 800])

phi = [1 0 0 0 0 0 0 0 0 1 0 0 1;
    0 1 0 0 0 0 0 0 0 1 0 0 1;
    0 0 1 0 0 0 0 0 0 1 0 0 1;
    0 0 0 1 0 0 0 0 0 0 1 0 1;
    0 0 0 0 1 0 0 0 0 0 1 0 1;
    0 0 0 0 0 1 0 0 0 0 1 0 1;
    0 0 0 0 0 0 1 0 0 0 0 1 1;
    0 0 0 0 0 0 0 1 0 0 0 1 1;
    0 0 0 0 0 0 0 0 1 0 0 1 1;]';
% view weights for chunkings
figure; hold on; colormap(brewermap([],'Reds'))
action = {'up','down','left','right'};
for i = 1:length(simdata)
    subplot(1,length(simdata),i); hold on;
    %imagesc(simdata(i).theta);
    %set(gca,'YDir','reverse')
    simdata(i).theta(simdata(i).theta<0) = 0;
    for s = 1:9
        idx = find(phi(:,s)==1);
        if ~isempty(find(simdata(i).theta(idx,:)<0))
            [ix,iy]=find(simdata(i).theta<0);
        end
        theta = simdata(i).theta(idx,:);
        [r,b]=find(max(theta));
        nTheta = theta./nansum(theta(:)); % normalize thetas
        pChunk(s,:) = nansum(nTheta,2)'; % sum over actions, percentage that chunk c contributes to policy for state s
        chunks(:,:,s) = nTheta;
    end
    
    barwitherr(sem(pChunk,1),mean(pChunk))
    title(strcat('\beta=',num2str(beta(i))))
    ylabel('% \theta contribution to policy')
    xlabel('chunking')
    nRew(i) = sum(sum(simdata(i).reward,2)./sum(simdata(i).action~=0,2))/size(simdata(i).action,1); % reward normalized by amount of steps it took
    pComplex(i) = mean(simdata(i).KL);
end
equalabscissa(1, length(simdata))
set(gcf, 'Position',  [500, 100, 1000, 300])

% normalized reward (by number of actions) and policy cost
rmap = plmColors(length(simdata),'r');
figure; hold on;
for i = 1:length(simdata)
    %subplot(1,length(simdata),i); hold on;
    %plot(simdata(i).KL,simdata(i).V,'o','Color',rmap(i,:),'MarkerSize',10)
    plot(pComplex(i),nRew(i),'.','Color',rmap(i,:),'MarkerSize',50)
end
l = legend(string(beta));
title(l,'\beta')
ylabel('Average reward')
xlabel('Policy complexity')




why
%% test

% look at the trajectory in the test world
% normalized reward (by number of actions) and policy cost

%         figure; hold on; colormap(brewermap([],'Reds'))
%         action = {'up','down','left','right'};
%         for i = 1:length(test)
%             subplot(1,length(test),i); hold on;
%
%             nSeq(i,:) = sum(test(i).action~=0,2) % number of actions it took to reach goal
%             aSeq(i,:) = test(i).state; % action sequence
%
%             nRew(i) = sum(sum(test(i).reward,2)./sum(test(i).action~=0,2))/size(test(i).action,1); % reward normalized by amount of steps it took
%             pComplex(i) = mean(test(i).KL);
%         end



end