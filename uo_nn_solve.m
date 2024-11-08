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

Xtr= 0; ytr= 0; wo= 0; fo= 0; tr_acc= 0; Xte= 0; yte= 0; te_acc= 0; niter= 0; tex= 0;

fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')
fprintf('[uo_nn_solve] Pattern recognition with neural networks.\n')
fprintf('[uo_nn_solve] %s\n',datetime)
fprintf('[uo_nn_solve] :::::::::::::::::::::::::::::::::::::::::::::::::::\n')

%
% Generate training data set
%
fprintf('[uo_nn_solve] Training data set generation.\n')
[Xtr,ytr]=uo_nn_dataset(te_seed,tr_p,num_target,tr_freq);
% uo_nn_Xyplot(Xtr,ytr,[])



%
% Generate test data set
%
fprintf('[uo_nn_solve] Test data set generation.\n');

[Xte,yte]=uo_nn_dataset(tr_seed,te_q,num_target,0);
% uo_nn_Xyplot(Xte,yte,[])


%
% Optimization
%


w0= rand(size(Xtr, 1), 1);
fprintf('[uo_nn_solve] Optimization\n');

if isd==1
    % Run Gradient Method (GM) to find w*
    [w_opt,iout] = GM(Xtr, ytr, w0,la,epsG,kmax,ils,ialmax, kmaxBLS,epsal,c1,c2);

elseif isd==2
    % Run Quasi-Newton Method (QNM) to find w*
    %w_opt = QNM(Xtr, ytr, w0, lambda, epsG, kmax, c1, c2, alpha_max);

else 
    %Run Stochastic Gradient Method (SGM) to find w*

end
fprintf('HOLA');



%
% Training accuracy
%
fprintf('[uo_nn_solve] Training Accuracy.\n');


uo_nn_Xyplot(Xtr,ytr,[w_opt]);
sigmoid = @(z) 1 ./ (1 + exp(-z));
y_pred_train = sigmoid(w_opt' * sigmoid(Xtr));  % Use w_opt_GM or w_opt_QNM
y_pred_train = y_pred_train >= 0.5;  % Threshold at 0.5 for binary classification

% Calculate Training Accuracy
accuracy_train = mean(y_pred_train == ytr) * 100;



%
% Test accuracy
%
fprintf('[uo_nn_solve] Test Accuracy.\n');

uo_nn_Xyplot(Xte,yte,[w_opt]);
y_pred_test = sigmoid(w_opt' * sigmoid(Xte)); 
y_pred_test = y_pred_test >= 0.5;  % Threshold at 0.5 for binary classification

% Calculate Test Accuracy
accuracy_test = mean(y_pred_test == yte) * 100;



%Xtr= Xtr;
%ytr=ytr;
wo=w_opt;
%fo?
tr_acc=accuracy_train;
% Xte=Xte;
% yte=yte;
te_acc=accuracy_test;
niter=iout;
tex=toc;

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End Procedure uo_nn_solve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
