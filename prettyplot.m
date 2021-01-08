function prettyplot(varargin)
% purpose: makes things aesthetically pleasing
% written by: lucy lai

%varargin: 1-fontsize
for i = 1:length(varargin)
    if i ==1
        fz = varargin{1};
    elseif i ==2
    end
end

% nice defaults
set(0, 'DefaultAxesFontName', 'Palatino');
set(0, 'DefaultTextFontName', 'Palatino');
set(0, 'DefaultAxesBox','off')
set(0, 'DefaultAxesLineWidth', 1.5)
set(0, 'DefaultAxesTickDir', 'out');
set(0, 'DefaultAxesTickDirMode', 'manual');
set(0, 'DefaultAxesTickDirMode', 'manual');
set(0, 'DefaultFigureColor',[1 1 1])
set(0, 'DefaultAxesFontSize',15)
set(0, 'DefaultLineLineWidth', 2) % set line width
set(0,'DefaultLegendAutoUpdate','off')
set(0,'DefaultAxesBox','off')
set(0,{'DefaultAxesXColor','DefaultAxesYColor','DefaultAxesZColor'},{'k','k','k'})
set(0,'defaultAxesFontSize',17)
% fontsize
if exist('fz')
    set(gca,'FontSize',fz); % make font bigger (input can be font size)
    box off
end

end