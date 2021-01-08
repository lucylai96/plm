function plot_figures(fig,results)
% psychology of learning and memory chapter
close all
prettyplot

switch fig
    
    %% perseveration: collins 2018
    case 'fig2'
        
        % main results
        if nargin < 2; load results_collins18.mat; end % else...need to run results = analyze_collins.mat
        C = linspecer(2);
        R = squeeze(nanmean(results.R));
        V = squeeze(nanmean(results.V));
        ylim = [0.25 1.1];
        xlim = [0 0.9];
        figure; hold on;
        for i = 1:2
            h(i) = plot(R(:,i),V(:,i),'LineWidth',4,'Color',C(i,:));
        end
        xlabel('Policy complexity','FontSize',25);
        ylabel('Average reward','FontSize',25);
        set(gca,'FontSize',25,'YLim',ylim,'XLim',xlim);
        
        for i = 1:2
            h(i+2) = plot(results.R_data(:,i),results.V_data(:,i),'o','Color',C(i,:),'MarkerSize',10,'LineWidth',3,'MarkerFaceColor',C(i,:));
        end
        legend(h,{'Ns = 3 (theory)' 'Ns = 6 (theory)' 'Ns = 3 (data)' 'Ns = 6 (data)'},'FontSize',20,'Location','SouthEast');
        legend('boxoff')
        prettyplot(25)
        
        % example subject(s):
        % for set size = 3
        % < 0.2 avg complexity subj = 13 (block 4 and block 10)
        % > 0.5 avg complexity subj = 31 (block 1, 10, 12), 29 (block 14)
        
        subj = [13 31]; blk = [4 1]; % [4, 10] [10 12]
        for i = 1:length(subj)
            ex_subj(i) = analyze_collins2(subj(i),blk(i));
            plot(results.R_data(subj(i),1),results.V_data(subj(i),1),'o','Color','k','MarkerSize',15,'LineWidth',3,'MarkerFaceColor',C(1,:));
            plot(results.R_data(subj(i),1),results.V_data(subj(i),1),'o','Color','k','MarkerSize',15,'LineWidth',3,'MarkerFaceColor',C(1,:));
        end
        
        exportgraphics(gcf,[pwd '/figures/' fig 'main.pdf'])
        
        for i = 1:length(subj)
            figure; hold on;
            P  = [ex_subj(i).Pa; ex_subj(i).Pas];
            for p = 1:size(P,1)
                subplot(size(P,1),1,p); hold on;
                bar(P(p,:));
                if p >1 ylabel(strcat('p(a|s_',num2str(p-1),')')); title(num2str(ex_subj(i).KL(p-1,:))); else ylabel('p(a)'); title(num2str(mean(ex_subj(i).KL))); end
                axis([0.3 3.7 0 1])
            end
            equalabscissa(size(P,1),1)
            xlabel('Action')
            set(gcf, 'Position',  [100, 100, 300, 700])
            exportgraphics(gcf,[pwd '/figures/fig2_ex' num2str(i) '.pdf'])
            %suptitle(num2str(results.R_data(subj(i),1)))
        end
        
        % steyvers example
        %         figure;
        %         if nargin < 2; load results_steyvers19.mat; end
        %         C = linspecer(1);
        %         plot(results.R,results.V,'LineWidth',4,'Color',C);
        %         hold on;
        %         plot(results.R_data,results.V_data,'o','MarkerSize',10,'LineWidth',3,'Color',C,'MarkerFaceColor',C);
        %         set(gca,'FontSize',25);
        %         xlabel('Policy complexity','FontSize',25);
        %         ylabel('Average reward','FontSize',25);
        %         legend({'theory' 'data'},'FontSize',25,'Location','East');
        
        
        
        %% perseveration: rule reversal
    case 'fig3'
        
        
        agent.lrate_V = .3;
        agent.lrate_p = .1;
        agent.lrate_theta = .3;
        beta = [0.1 1 1.5 2 3]; % capacity constraint
        %beta = 1;
        map = plmColors(length(beta),'b');
        
        figure; hold on;
        for i = 1:length(beta)
            subplot 311; hold on;
            agent.beta = beta(i);
            simdata(i) = sim_revlearn(agent);
            pCorr = [reshape(simdata(i).corchoice(simdata(i).trueS==1),simdata(i).tpb,simdata(i).nRevs/2)'; reshape(simdata(i).corchoice(simdata(i).trueS==2),simdata(i).tpb,simdata(i).nRevs/2)'];
            errorbar(0:simdata(i).tpb-1,mean(pCorr),sem(pCorr,1),'o-','Color',map(i,:),'LineWidth',2)
            
            subplot 313; hold on;
            bar(i,simdata(i).belief,'FaceColor',map(i,:))
        end
        
        subplot 311;
        l = legend(string(beta));
        title(l,'\beta')
        axis([0 15 0 1])
        ylabel('p(Correct)')
        xlabel('trials after reversal')
        prettyplot(18)
        
        subplot 312;
        bar([[simdata.losestay];[simdata.losestay]])
        xticks([1:2])
        set(gca, 'XTickLabel', {'Lose-Stay','Win-Shift'})
        ylabel('% of trials')
        l = legend(string(beta));
        legend('boxoff')
        title(l,'\beta')
        box off;
        prettyplot(18)
        
        subplot 313;
        xticks([1:5])
        set(gca, 'XTickLabel', num2cell(beta))
        xlabel('\beta')
        ylabel('p(true state = belief state)')
        prettyplot(18)
        box off;
        
        set(gcf, 'Position',  [600, 50, 500, 800])
        why
        %% state chunking: contextual bandits
    case 'fig4'
        
        % a = 1 is left, a = 2 is right
        % s = 1 is A, s = 2 is B
        %     A1   A2
        % S1  1     0
        % S2  1     0
        
        %     A1   A2
        % S1  1     0
        % S2  0     1
        
        agent.lrate_V = 0.1;
        agent.lrate_p = 0.01;
        agent.lrate_theta = 0.1;
        beta = [0.1 0.5 1 1.5 2]; % capacity constraint
        
        %beta = 0:0.5:3;
        
        %figure; hold on;
        for i = 1:length(beta)
            agent.beta = beta(i);
            simdata(i) = sim_bandit(agent);
            
            % if state chunking, show that p(A2|S1) increases along with p(A2|S2)
            rev = length(simdata(i).state)/2; % reversal
            A2S1 = simdata(i).action==2 & simdata(i).state==1;
            pA2S1(i,:) = [sum(A2S1(1:rev-1)) sum(A2S1(rev:end))]./(rev/2);
            dpA2S1(i) = pA2S1(i,2)-pA2S1(i,1);
            
            A2S2 = simdata(i).action==2 & simdata(i).state==2;
            pA2S2(i,:) = [sum(A2S2(1:rev-1)) sum(A2S2(rev:end))]./(rev/2);
            dpA2S2(i) = pA2S2(i,2)-pA2S2(i,1);
            
            %             subplot(1,length(beta),i); hold on;
            %             var = [simdata(i).pas(:,:,2)-simdata(i).pas(:,:,1)];
            %             pdiff2(:,:,i) = var;
            %             imagesc(var)
            %             set(gca,'YDir','reverse')
            %             title(strcat('\beta=',num2str(beta(i))))
        end
        %eqcolorbar(1,length(beta))
        
        
        bmap = plmColors(length(beta),'b');
        gmap = plmColors(length(beta),'g');
        
        figure; hold on;
        subplot 221; hold on;
        %ylim([0 0.5])
        h = bar(beta,pA2S1); xlabel('\beta'); ylabel('p(choose A_2|S_1)');
        legend('block 1','block 2'); legend('boxoff')
        set(gca, 'XTickLabel', num2cell(beta))
        box off
        set(h(1),'facecolor',bmap(1,:))
        set(h(2),'facecolor',bmap(3,:))
        
        subplot 222; hold on;
        bar(beta,dpA2S1,'facecolor',bmap(2,:)); xlabel('\beta'); ylabel('\Delta p(choose A_2|S_1)');
        set(gca, 'XTickLabel', num2cell(beta))
        
        
        subplot 223; hold on;
        h = bar(beta,pA2S2); xlabel('\beta'); ylabel('p(choose A_2|S_2)');
        legend('block 1','block 2'); legend('boxoff')
        set(gca, 'XTickLabel', num2cell(beta))
        set(h(1),'facecolor',gmap(1,:))
        set(h(2),'facecolor',gmap(3,:))
        box off
        
        subplot 224; hold on;
        h = bar(beta,dpA2S2,'facecolor',gmap(2,:)); xlabel('\beta'); ylabel('\Delta p(choose A_2|S_2)');
        set(gca, 'XTickLabel', num2cell(beta))
        
        set(gcf, 'Position',  [100, 100, 800, 400])
        
        exportgraphics(gcf,[pwd '/figures/fig4_ex1.pdf'])
        
        
        figure; hold on;
        for i = 1:length(beta)
            subplot 321; hold on;
            bar(i, simdata(i).pa(rev,:),'k')
            ylabel('p(A)');  set(gca, 'XTickLabel', num2cell(beta))
            title('end of block 1')
            
            subplot 322; hold on;
            bar(i,simdata(i).pa(end,:),'k'); set(gca, 'XTickLabel', num2cell(beta))
            title('end of block 2')
            
            subplot 323; hold on;
            bar(i,simdata(i).pas(1,:,1),'FaceColor', bmap(i,:)); % p(A|S1) block 1
            set(gca, 'XTickLabel', num2cell(beta))
            ylabel('p(A|S_1)')
            
            subplot 324; hold on;
            bar(i,simdata(i).pas(1,:,2),'FaceColor', bmap(i,:)); % p(A|S1) block 2
            set(gca, 'XTickLabel', num2cell(beta))
            
            subplot 325; hold on;
            bar(i,simdata(i).pas(2,:,1),'FaceColor', gmap(i,:)); % p(A|S2) block 1
            set(gca, 'XTickLabel', num2cell(beta))
            ylabel('p(A|S_2)')
            xlabel('\beta')
            
            subplot 326; hold on;
            bar(i,simdata(i).pas(2,:,2),'FaceColor', gmap(i,:)); % p(A|S2) block 2
            set(gca, 'XTickLabel', num2cell(beta))
            xlabel('\beta')
        end
        equalabscissa(3,2)
        set(gcf, 'Position',  [100, 100, 800, 500])
        
        figure; hold on;
        subplot 121; hold on;
        for s = 1:2 % 2 states
            map(:,:,1) = bmap;
            map(:,:,2) = gmap;
            for i = 1:length(beta)
                plot(simdata(i).KL(s,1),simdata(i).avgrw(s,1),'.','Color',map(i,:,s),'MarkerSize',40)
            end
        end
        ylabel('Average reward')
        xlabel('Policy complexity')
        title('end of block 1')
        
        subplot 122; hold on;
        for s = 1:2 % 2 states
            for i = 1:length(beta)
                plot(simdata(i).KL(s,2),simdata(i).avgrw(s,2),'.','Color',map(i,:,s),'MarkerSize',40)
            end
        end
        ylabel('Average reward')
        xlabel('Policy complexity')
        title('end of block 2')
        
        equalabscissa(1,2)
        set(gcf, 'Position',  [100, 100, 700, 300])
        
        % plot([simdata.pa]); legend('S_1','S_2'); ylabel('p(A)')
        
        
        %% state chunking: gridworld
    case 'fig5'
        
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
        
        
        
        %% action chunking
    case 'fig6'
        
    case 'fig7'
        %% navigation
        
        
        
        
end

end

function lineStyles=linspecer(N,varargin)

if nargin==0 % return a colormap
    lineStyles = linspecer(64);
    %     temp = [temp{:}];
    %     lineStyles = reshape(temp,3,255)';
    return;
end

if N<=0 % its empty, nothing else to do here
    lineStyles=[];
    return;
end

% interperet varagin
qualFlag = 0;

if ~isempty(varargin)>0 % you set a parameter?
    switch lower(varargin{1})
        case {'qualitative','qua'}
            if N>12 % go home, you just can't get this.
                warning('qualitiative is not possible for greater than 12 items, please reconsider');
            else
                if N>9
                    warning(['Default may be nicer for ' num2str(N) ' for clearer colors use: whitebg(''black''); ']);
                end
            end
            qualFlag = 1;
        case {'sequential','seq'}
            lineStyles = colorm(N);
            return;
        otherwise
            warning(['parameter ''' varargin{1} ''' not recognized']);
    end
end

% predefine some colormaps
set3 = colorBrew2mat({[141, 211, 199];[ 255, 237, 111];[ 190, 186, 218];[ 251, 128, 114];[ 128, 177, 211];[ 253, 180, 98];[ 179, 222, 105];[ 188, 128, 189];[ 217, 217, 217];[ 204, 235, 197];[ 252, 205, 229];[ 255, 255, 179]}');
set1JL = brighten(colorBrew2mat({[228, 26, 28];[ 55, 126, 184];[ 77, 175, 74];[ 255, 127, 0];[ 255, 237, 111]*.95;[ 166, 86, 40];[ 247, 129, 191];[ 153, 153, 153];[ 152, 78, 163]}'));
set1 = brighten(colorBrew2mat({[ 55, 126, 184]*.95;[228, 26, 28];[ 77, 175, 74];[ 255, 127, 0];[ 152, 78, 163]}),.8);

set3 = dim(set3,.93);

switch N
    case 1
        lineStyles = { [  55, 126, 184]/255};
    case {2, 3, 4, 5 }
        lineStyles = set1(1:N);
    case {6 , 7, 8, 9}
        lineStyles = set1JL(1:N)';
    case {10, 11, 12}
        if qualFlag % force qualitative graphs
            lineStyles = set3(1:N)';
        else % 10 is a good number to start with the sequential ones.
            lineStyles = cmap2linspecer(colorm(N));
        end
    otherwise % any old case where I need a quick job done.
        lineStyles = cmap2linspecer(colorm(N));
end
lineStyles = cell2mat(lineStyles);
end

% extra functions
function varIn = colorBrew2mat(varIn)
for ii=1:length(varIn) % just divide by 255
    varIn{ii}=varIn{ii}/255;
end
end

function varIn = brighten(varIn,varargin) % increase the brightness

if isempty(varargin),
    frac = .9;
else
    frac = varargin{1};
end

for ii=1:length(varIn)
    varIn{ii}=varIn{ii}*frac+(1-frac);
end
end

function varIn = dim(varIn,f)
for ii=1:length(varIn)
    varIn{ii} = f*varIn{ii};
end
end

function vOut = cmap2linspecer(vIn) % changes the format from a double array to a cell array with the right format
vOut = cell(size(vIn,1),1);
for ii=1:size(vIn,1)
    vOut{ii} = vIn(ii,:);
end
end
%%
% colorm returns a colormap which is really good for creating informative
% heatmap style figures.
% No particular color stands out and it doesn't do too badly for colorblind people either.
% It works by interpolating the data from the
% 'spectral' setting on http://colorbrewer2.org/ set to 11 colors
% It is modified a little to make the brightest yellow a little less bright.
function cmap = colorm(varargin)
n = 100;
if ~isempty(varargin)
    n = varargin{1};
end

if n==1
    cmap =  [0.2005    0.5593    0.7380];
    return;
end
if n==2
    cmap =  [0.2005    0.5593    0.7380;
        0.9684    0.4799    0.2723];
    return;
end

frac=.95; % Slight modification from colorbrewer here to make the yellows in the center just a bit darker
cmapp = [158, 1, 66; 213, 62, 79; 244, 109, 67; 253, 174, 97; 254, 224, 139; 255*frac, 255*frac, 191*frac; 230, 245, 152; 171, 221, 164; 102, 194, 165; 50, 136, 189; 94, 79, 162];
x = linspace(1,n,size(cmapp,1));
xi = 1:n;
cmap = zeros(n,3);
for ii=1:3
    cmap(:,ii) = pchip(x,cmapp(:,ii),xi);
end
cmap = flipud(cmap/255);
end