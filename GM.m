function [w_opt,iout]=GM(Xtr,ytr,w0,la,epsG, kmax,ils,ialmax,kmaxBLS,epsal, c1, c2)
    alpha_max= 1.0; 
    sig = @(Xtr) 1./(1+exp(-Xtr));
    y = @(Xtr,w ) sig (w'*sig(Xtr));
    L = @(w,Xtr,ytr) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2) + (la*norm (w)^2)/2;
    gL = @(w,Xtr,ytr) (2*sig(Xtr)*((y(Xtr, w)-ytr) .*y(Xtr, w) .* (1-y(Xtr,w)))')/size (ytr,2)+la*w;

    w = w0;
    for k = 1:kmax
        grad = gL(w, Xtr, ytr);
        if norm(grad) < epsG
            break;
        end
        if ils==1
        elseif ils==2
        else
            [alpha, iout] = uo_BLSNW32(@(w) L(w, Xtr, ytr), @(w) gL(w, Xtr, ytr), w, -grad, alpha_max, c1, c2, kmaxBLS, epsal,ialmax);
        end
        w = w - alpha * grad;
    end
    w_opt = w;

end


