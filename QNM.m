function [w_opt, iout] = QNM(Xtr, ytr, w0, la, epsG, kmax, ils, ialmax, kmaxBLS, epsal, c1, c2)
    alpha_max = 1;
    sig = @(Xtr) 1./(1+exp(-Xtr));
    y = @(Xtr,w) sig(w'*sig(Xtr));
    L = @(w,Xtr,ytr) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2) + (la*norm(w)^2)/2;
    gL= @(w,Xtr,ytr) (2*sig(Xtr)*((y(Xtr,w)-ytr).*y(Xtr,w).*(1-y(Xtr,w)))')/size(ytr,2)+la*w;


    w = w0;
    I=eye(length(w0));
    H = I;  % Initial inverse Hessian approximation
    grad = gL(w, Xtr, ytr);
    for k = 1:kmax
        if norm(grad) < epsG
            break;
        end
        
        % Compute search direction
        d = -H * grad;
        
        % Perform line search to find step size alpha
        [alpha, ~] = uo_BLSNW32(@(w) L(w, Xtr, ytr), @(w) gL(w, Xtr, ytr), w, d, alpha_max, c1, c2, kmaxBLS, epsal, ialmax);
        
        % Update weights
        w_new = w + alpha * d;
        
        % Compute new gradient
        grad_new = gL(w_new, Xtr, ytr);
        
        % Update the inverse Hessian matrix H using the BFGS formula
        s = w_new - w;
        y = grad_new - grad;
        rho = 1 / (y' * s);
        
        % BFGS update to H
        H = (I - rho * s * y') * H * (I - rho * y * s') + rho * s * s';
        
        % Move to new weights for the next iteration
        w = w_new;
        grad=grad_new;
    end
    
    % Final output assignment
    w_opt = w;
    iout = k;






 % 
 % 
 % % Quasi Newton with BFGS
 %    I = eye(length(w));
 %    h=I;
 %    %wantic = -1;
 %    %fprintf("%d\n", size(gL(w,Xtr,ytr)));
 %    while norm(gL(w, Xtr, ytr)) >= epsG && k <= kmax
 % 
 %        d = -(h*gL(w, Xtr, ytr));
 % 
 %        if k > 1
 %           % ialmax = alk(k-1)*(dot(gL(xk(:,k-1), Xtr, ytr), dk(:,k-1)))/ (dot(gL(w, Xtr, ytr), d));
 %           almax = (2*(L(w, Xtr, ytr)-L(xk(:,k-1), Xtr, ytr)))/(dot(gL(w, Xtr, ytr), d));
 %        else 
 %            almax = 1;
 % 
 %        end
 % 
 %        [al,~] = uo_BLSNW32(L,gL,w, Xtr, ytr,d, almax,c1,c2,kmaxBLS,epsal);
 % 
 %        wantic = w;
 %        w = w + al*d;
 %        k = k+1;
 % 
 % 
 %        s = w-wantic ;
 %        yk = gL(w, Xtr, ytr)- gL(wantic, Xtr, ytr);
 %        rh = 1 / (yk'*s);
 % 
 %        h = (I - rh*s*yk')*h*(I-rh*yk*s') + rh*s*s';
 % 
 %        xk = [xk,w]; 
 %        alk = [alk, al];
 %        % update all param:
 %        %xk = [xk,x]; fk = [fk,f(x)]; gk = [gk,g(x)];  Hk{end+1} = h;
 % 
 %    end
 % 
 %    wo = w;
 % 
 % 

end
