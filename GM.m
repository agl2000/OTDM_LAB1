function [w_opt,iout]=GM(Xtr,ytr,w0,la,epsG, kmax,ils,ialmax,kmaxBLS,epsal, c1, c2)
    alpha_max= 1.0; 
    sig = @(Xtr) 1./(1+exp(-Xtr));
    y = @(Xtr,w) sig (w'*sig(Xtr));
    L = @(w,Xtr,ytr) (norm(y(Xtr,w)-ytr)^2)/size(ytr,2) + (la*norm(w)^2)/2;
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
            [alpha, ~] = uo_BLSNW32(@(w) L(w, Xtr, ytr), @(w) gL(w, Xtr, ytr), w, -grad, alpha_max, c1, c2, kmaxBLS, epsal,ialmax);
        end
        
        w = w - alpha * grad;
    end
    w_opt = w;
    iout=k;

end

% %%% Gradient Method
%     k=1;
%     w=w0;
%     xk=[];
%     alk=[];
%     while norm(gL(w, Xtr, ytr)) >= epsG && k <= kmax
% 
%         d = -gL(w, Xtr, ytr); 
% 
%         % dk = [dk, d];
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
%         [al,~] = uo_BLSNW32(@(w) L(w, Xtr, ytr), @(w) gL(w, Xtr, ytr), w, d, almax,c1,c2,kmaxBLS,epsal);
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
%     w_opt = w;
%     iout=k;
% 
% end
% 

