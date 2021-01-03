function plm1
% showing action choices for a simple 2 action task

addpath('/Users/lucy/Google Drive/Harvard/Projects/mat-tools')


A = 2; % num actions
S = 10000; % num states

Ps = repmat(1/S, 1, S); % p(states)
Q = [rand(S,A)]; % values of each action are independently uniformly random between 0 and 1
Q(:,1) = 2*Q(:,1); % choice A is on avg 2x more valuable
beta = [1, 4, 100]; 
beta = [1 2 5 10 50];

[R,V,Pa,Psa] = blahut_arimoto2(Ps,Q,beta);

vdiff = (Q(:,1) - Q(:,2))'; % value difference between action 1 and action 2

%choiceM = [Psa(1,:,1); Psa(2,:,1); Psa(3,:,1); vdiff]'; % 3 beta values x choose A
choiceM  = [Psa(:,:,1); vdiff]';
choiceMsort = sortrows(choiceM, size(choiceM,2)); % sort 
vdiff = choiceMsort(:,end);

figure; hold on;
map = colormap(brewermap(size(choiceM,2),'Blues'));
map = map(2:end,:);
set(0, 'DefaultAxesColorOrder', map) % first three rows
for i = 1:size(choiceM,2)-1
    plot(vdiff, choiceMsort(:,i),'LineWidth', 3)
end 

%plot(vdiff, choiceMsort(:,1),vdiff, choiceMsort(:,2), vdiff, choiceMsort(:,3), 'LineWidth', 3)


axis([-1 1 0 1])
xlabel('Value difference (A-B)')
ylabel('P(choose A)')
lgd = legend('Low \beta', 'Medium \beta', 'High \beta');
legend('boxoff')
lgd.Location = 'southeast';
prettyplot
axis square
%scatter(Qrewards(:,1) - Qrewards(:,2), Psa(1,:,1))
%hold on
%scatter(Qrewards(:,1) - Qrewards(:,2), Psa(2,:,1))
%scatter(Qrewards(:,1) - Qrewards(:,2), Psa(3,:,1))
end