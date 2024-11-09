function [w_opt,iout]=QNM(Xtr, ytr, w0, la, epsG, kmax,ils,ialmax,kmaxBLS,epsal, c1, c2)
    alpha_max= 1.0; 
    sig = @(Xtr) 1./(1+exp(-Xtr));
    y = @(Xtr,w ) sig (w'*sig(Xtr));
    L = @(w,Xtr,ytr) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2) + (la*norm (w)^2)/2;
    gL = @(w,Xtr,ytr) (2*sig(Xtr)*((y(Xtr, w)-ytr) .*y(Xtr, w) .* (1-y(Xtr,w)))')/size (ytr,2)+la*w;    

    w = w0;
    H = eye(length(w0));  % Initial inverse Hessian approximation
    for k = 1:kmax
        grad = gL(w, Xtr, ytr);
        if norm(grad) < epsG
            break;
        end
        d = -H * grad;  % Quasi-Newton search direction
        [alpha, iout] = uo_BLSNW32(@(w) L(w, Xtr, ytr), @(w) gL(w, Xtr, ytr), w, d, alpha_max, c1, c2, kmaxBLS, epsal, ialmax);
        w_new = w + alpha * d;
        grad_new = gL(w_new, Xtr, ytr);
        
        % Update H using BFGS
        s = w_new - w;
        y = grad_new - grad;
        rho = 1 / (y' * s);
        H = (eye(length(w0)) - rho * s * y') * H * (eye(length(w0)) - rho * y * s') + rho * s * s';
        
        w = w_new;
    end
    w_opt = w;
end