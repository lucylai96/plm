function plot_figures(fig,results)
% psychology of learning and memory chapter
prettyplot

switch fig
    
    % illustrative figure, collins 2018
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

        
    case 'fig3'
        
        
    case 'fig4'
        
   
    case 'fig5'
        
 
        
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