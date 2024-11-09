%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% F.-Javier Heredia https://gnom.upc.edu/heredia
% Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Xtr,ytr,wo,fo,tr_acc,Xte,yte,te_acc,niter,tex]=uo_nn_solve(num_target,tr_freq,tr_seed,tr_p,te_seed,te_q,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2,isd,sg_al0,sg_be,sg_ga,sg_emax,sg_ebest,sg_seed,icg,irc,nu)

tic

% Input parameters:
%
% num_target : set of digits to be identified.
%    tr_freq : frequency of the digits target in the data set.
%    tr_seed : seed for the training set random generation.
%       tr_p : size of the training set.
%    te_seed : seed for the test set random generation.
%       te_q : size of the test set.
%         la : coefficient lambda of the decay factor.
%       epsG : optimality tolerance.
%       kmax : maximum number of iterations.
%        ils : line search (1 if exact, 2 if uo_BLS, 3 if uo_BLSNW32)
%     ialmax :  formula for the maximum step lenght (1 or 2).
%    kmaxBLS : maximum number of iterations of the uo_BLSNW32.
%      epsal : minimum progress in alpha, algorithm up_BLSNW32
%      c1,c2 : (WC) parameters.
%        isd : optimization algorithm.
%     sg_al0 : \alpha^{SG}_0.
%      sg_be : \beta^{SG}.
%      sg_ga : \gamma^{SG}.
%    sg_emax : e^{SGÇ_{max}.
%   sg_ebest : e^{SG}_{best}.
%    sg_seed : seed for the first random permutation of the SG.
%        icg : if 1 : CGM-FR; if 2, CGM-PR+      (useless in this project).
%        irc : re-starting condition for the CGM (useless in this project).
%         nu : parameter of the RC2 for the CGM  (useless in this project).
%
% Output parameters:
%
%    Xtr : X^{TR}.
%    ytr : y^{TR}.
%     wo : w^*.
%     fo : {\tilde L}^*.
% tr_acc : Accuracy^{TR}.
%    Xte : X^{TE}.
%    yte : y^{TE}.
% te_acc : Accuracy^{TE}.
%  niter : total number of iterations.
%    tex : total running time (see "tic" "toc" Matlab commands).
%

% find and minimize L(w,Xtr, ytr, lambda)
sig = @(Xtr) 1./(1+exp(-Xtr));
y = @(Xtr,w) sig(w'*sig(Xtr));
L = @(w,Xtr,ytr) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2) + (la*norm(w)^2)/2;
gL= @(w,Xtr,ytr) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;


%Xtr= 0; ytr= 0; wo= 0; fo= 0; tr_acc= 0; Xte= 0; yte= 0; te_acc= 0; niter= 0; tex= 0;


fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')
fprintf('[uo_nn_solve] Pattern recognition with neural networks.\n')
fprintf('[uo_nn_solve] %s\n',datetime)
fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')





%
% Generate training data set
%
fprintf('[uo_nn_solve] Training data set generation.\n')
[Xtr,ytr] = uo_nn_dataset(tr_seed, tr_p, num_target, tr_freq);
% uo_nn_Xyplot(Xtr,ytr,[])



%
% Generate test data set
%
fprintf('[uo_nn_solve] Test data set generation.\n');
[Xte, yte] = uo_nn_dataset(te_seed,te_q, num_target, 0);

% uo_nn_Xyplot(Xte,yte,[])


%
% Optimization
%


w0= rand(size(Xtr, 1), 1);
fprintf('[uo_nn_solve] Optimization\n');

if isd==1
    fprintf('Run Gradient Method (GM) to find w*\n')
    [w_opt,iout] = GM(Xtr, ytr, w0,la,epsG,kmax,ils,ialmax, kmaxBLS,epsal,c1,c2);
    fo=L(w_opt, Xtr, ytr);

elseif isd==2
    fprintf('Run Quasi-Newton Method (QNM) to find w*\n')
    [w_opt,iout] = QNM(Xtr,ytr,w0,la,epsG,kmax,ils,ialmax,kmaxBLS,epsal,c1,c2);
    fo=L(w_opt, Xtr, ytr);

else 
    fprintf('Run Stochastic Gradient Method (SGM) to find w*\n')
    % Stochastic gradient method
    [alef, bet] = size(Xtr);
    p = bet;
    m = floor(p*sg_ga);
    %sg_k = ceil(p/m);
    sg_kmax  = sg_emax;
    e = 0;
    s = 0;
    l_te_best = +Inf;
    k = 0;
    w = w0;
    sg_al = 0.01*sg_al0; % for learning rate ak
    sg_k = floor(sg_be*sg_kmax); %for learning rate ak

    while e <= sg_emax && s < sg_ebest

        pm = randperm(p);

        for i = 0 : ceil(p/m-1)
            S = pm(i*m+1:min((i+1)*m, p));
            X_tr_s = Xtr(:,S);
            
            y_tr_s = ytr(S);
            dk = -gL(w, X_tr_s, y_tr_s);
            

            if k <= sg_k
              al_k =   (1 - k / sg_k)*sg_al0 + (k / sg_k)*sg_al;
            elseif k > sg_k
                al_k = sg_al;
            end

            w = w + al_k*dk;
            k = k+1;
        end

        
        e = e+1;
        %("%f \n", wk);
        l_te = L(w, Xte, yte);
        %fprintf("%f %f\n", l_te, l_te_best);
        if l_te <  l_te_best
            l_te_best = l_te;
            w_opt = w;
            s = 0;
        else
            s = s+1;
        end 
        
    end 
    fo=L(w_opt, Xtr, ytr);
    iout=k;


end




%
% Training accuracy
%
fprintf('[uo_nn_solve] Training Accuracy.\n');



% uo_nn_Xyplot(Xtr,ytr,[w_opt]);
sigmoid = @(z) 1 ./ (1 + exp(-z));
y_pred_train = sigmoid(w_opt' * sigmoid(Xtr));  % Use w_opt_GM or w_opt_QNM
y_pred_train = y_pred_train >= 0.5;  % Threshold at 0.5 for binary classification

% Calculate Training Accuracy
accuracy_train = mean(y_pred_train == ytr) * 100;



%
% Test accuracy
%
fprintf('[uo_nn_solve] Test Accuracy.\n');

% uo_nn_Xyplot(Xte,yte,[w_opt]);
y_pred_test = sigmoid(w_opt' * sigmoid(Xte)); 
y_pred_test = y_pred_test >= 0.5;  % Threshold at 0.5 for binary classification

% Calculate Test Accuracy
accuracy_test = mean(y_pred_test == yte) * 100;



%Xtr= Xtr;
%ytr=ytr;
wo=w_opt;
%fo?
tr_acc=accuracy_train
% Xte=Xte;
% yte=yte;
te_acc=accuracy_test
niter=iout;
tex=toc;


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% 
% 
% %%% Gradient Method
%     while norm(gL(w, Xtr, ytr)) >= epsG && k <= kmax
% 
%         d = -gL(w, Xtr, ytr); 
% 
%         dk = [dk, d];
% 
%         if k > 1
%            % ialmax = alk(k-1)*(dot(gL(xk(:,k-1), Xtr, ytr), dk(:,k-1)))/ (dot(gL(w, Xtr, ytr), d));
%            almax = (2*(L(w, Xtr, ytr)-L(xk(:,k-1), Xtr, ytr)))/(dot(gL(w, Xtr, ytr), d));
%         else 
%             almax = 1;
%         end
%         %fprintf("%d\n", k);
% 
% 
% 
%         [al,iWv] = uo_BLSNW32(L,gL, w, Xtr, ytr, d, almax,c1,c2,kmaxBLS,epsal);
% 
% 
% 
%         w = w + al*d; k = k+1; % GM iteration
%         %fprintf("%f \n", w);
% 
%         % all the k+1 things
%         xk = [xk,w]; 
%         alk = [alk, al];
% 
% 
%     end
%     wo = w;
% 
% 
% 
% 
% 
% 
%     % Quasi Newton with BFGS
%     I = eye(length(w));
%     h=I;
%     wantic = -1;
%     %fprintf("%d\n", size(gL(w,Xtr,ytr)));
%     while norm(gL(w, Xtr, ytr)) >= epsG && k <= kmax
% 
%         d = -(h*gL(w, Xtr, ytr));
% 
%         if k > 1
%            % ialmax = alk(k-1)*(dot(gL(xk(:,k-1), Xtr, ytr), dk(:,k-1)))/ (dot(gL(w, Xtr, ytr), d));
%            almax = (2*(L(w, Xtr, ytr)-L(xk(:,k-1), Xtr, ytr)))/(dot(gL(w, Xtr, ytr), d));
%         else 
%             almax = 1;
% 
%         end
% 
%         [al,iWv] = uo_BLSNW32(L,gL,w, Xtr, ytr,d, almax,c1,c2,kmaxBLS,epsal);
% 
%         wantic = w;
%         w = w + al*d;
%         k = k+1;
% 
% 
%         s = w-wantic ;
%         yk = gL(w, Xtr, ytr)- gL(wantic, Xtr, ytr);
%         rh = 1 / (yk'*s);
% 
%         h = (I - rh*s*yk')*h*(I-rh*yk*s') + rh*s*s';
% 
%         xk = [xk,w]; 
%         alk = [alk, al];
%         % update all param:
%         %xk = [xk,x]; fk = [fk,f(x)]; gk = [gk,g(x)];  Hk{end+1} = h;
% 
%     end
% 
%     wo = w;
% 