function [lme_bias,lme_complexity] = bias_lme
    
    % Mixed effects regression analysis of bias.
    
    load results_collins14.mat
    data = load_data;
    
    cond = [data.cond];
    
    % run regression
    n = length(cond);
    for i=1:5
        setsize(:,i) = zeros(n,1)+i;
        sub(:,i) = (1:n)';
    end
    
    tbl = table;
    tbl.SZ = categorical(repmat(cond',5,1));
    tbl.setsize = categorical(setsize(:));
    tbl.bias = results.bias(:);
    tbl.complexity = results.R_data(:);
    tbl.sub = sub(:);
    
    lme_bias = fitlme(tbl,'bias ~ SZ*setsize + (setsize|sub)');
    anova(lme_bias)
    
    lme_complexity = fitlme(tbl,'complexity ~ SZ*setsize + (setsize|sub)');
    anova(lme_complexity)
    
    biasdiff_HC = results.bias(cond==0,end) - results.bias(cond==0,1);
    biasdiff_SZ = results.bias(cond==1,end) - results.bias(cond==1,1);
    [~,p,~,stat] = ttest2(biasdiff_HC,biasdiff_SZ);
    disp(['HC vs. SZ, setsize 6 - 2: t(',num2str(stat.df),') = ',num2str(stat.tstat),', p = ',num2str(p)]);
    
    [~,p,~,stat] = ttest(results.bias(cond==0,1),results.bias(cond==0,end));
    disp(['HC, setsize 2 vs. 6: t(',num2str(stat.df),') = ',num2str(stat.tstat),', p = ',num2str(p)]);
    [~,p,~,stat] = ttest(results.bias(cond==1,1),results.bias(cond==1,end));
    disp(['SZ, setsize 2 vs. 6: t(',num2str(stat.df),') = ',num2str(stat.tstat),', p = ',num2str(p)]);