function plmColors(n)
% n - number of colors
map = colormap(brewermap(n+2,'Blues'));
set(0, 'DefaultAxesColorOrder', map(3:end,:)) 

set(0, 'DefaultLineLineWidth', 3) % set line width