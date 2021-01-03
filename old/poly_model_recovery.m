function C = poly_model_recovery
    
    rep = 1000;
    s = 0.5;
    N = 85;
    x = linspace(0,1,N)';
    x = [ones(N,1) x x.^2];
    X = [x; x];
    bic = zeros(rep,2);
    aic = zeros(rep,2);
    
    for i = 1:rep
        for k = 1:2
            if k==1
                w = randn(3,1);
            end
            Y = [];
            for j = 1:2
                if k==2
                    w = randn(3,1);
                end
                y = x*w + randn(N,1)*s;
                Y = [Y; y];
                b = regress(y,x);
                mse = mean((y-x*b).^2);
                bic(i,k) = bic(i,k) + N*log(mse) + 3*log(N);
                aic(i,k) = aic(i,k) + N*log(mse) + 3*2;
            end
            b = regress(Y,X);
            mse = mean((Y-X*b).^2);
            bic(i,k) = bic(i,k) - N*2*log(mse) - 3*log(N*2);    % delta BIC - positive values favor joint model
            aic(i,k) = aic(i,k) - N*2*log(mse) - 3*2;
        end
    end
    
    %C.bic = mean(bic); C.aic = mean(aic);
    C.bic(1) = mean(bic(:,1)>0); C.bic(2) = mean(bic(:,2)<0);
    C.aic(1) = mean(aic(:,1)>0); C.aic(2) = mean(aic(:,2)<0);