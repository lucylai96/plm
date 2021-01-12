function map = plmColors(n,c)
% n - number of colors

switch c
    case 'b'
        map = colormap(brewermap(n+2,'Blues'));
        map = map(3:end,:);
        
    case 'g'
        map = colormap(brewermap(n+2,'Greens'));
        map = map(3:end,:);
    case'r'
        map = colormap(brewermap(n+2,'Reds'));
        map = map(3:end,:);
        
end

set(0, 'DefaultAxesColorOrder', map)
%set(gca,'ColorOrder','factory')

%set(0, 'DefaultLineLineWidth', 2) % set line width