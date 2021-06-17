function  [x,out]  =  r3gmresSubproblem(A,b,U,opts)
%   [x,out]  =  r3gmresSubproblem(A,b,U,opts)
%
% Implements an alternative version of the rrrGMRES algorithm proposed in: 
% Dong, Yiqiu, Henrik Garde, and Per Christian Hansen. 
% "R3GMRES: including prior information in GMRES-type methods 
% for discrete inverse problems." 
% Electronic Transactions on Numerical Analysis 42 (2014): 136-146.
%
% This code here and the approch to implementing an augmented code are
% describe in: 
%
% KM Soodhalter. ﻿A note on augmented unprojected Krylov subspace methods.
% Submitted for publication.
% 
%
% FUNCTION INFORMATIONS
% INPUTS:
%    A     nxn discrete ill-posed problem
%    b     noisy right-hand side
%    U     augmentation subspace
%    opts    Structure containing optional inputs or options
%    -opts.x0  initial approximation; default: zero vector
%    -opts.m   maximum number of iteration; default:100
%    -opts.tol convergence tolerance; default:1e-3
%    -opts.xtrue   true solution to check error
%    -opts.isrr      indicates to use range-restriction; default: true
%    -opts.checkTimeout   If the bound falsely told us we converge, this
%                         optional tells us how many iterations to do
%                         before we explicitely reconstruct the solution to
%                         check again; default: 5
% OUTPUTS:
%    x      Reconstructed solution
%    out    Structure containing optional outputs
%    -out.X    Solution history
%    -out.R    Residual history
%    -out.approxresvec   approximated residual norm upper bounds
%    -out.resvec    residual norms
%    -out.errvec    error norms
% OUTPUTS:
%    UpdateOrthBasisRank1QR  Produces a QR factorization of a rank-one
%                            update of a matrix with orthonormal columns
%    givens                  performs complex givens rotations
%
% Kirk M. Soodhalter
% 02-May-2021 18:46:57
% kirk@math.soodhalter.com

    if ~exist('opts','var')
        opts = struct(); %empty structure
    end

    if isfield(opts,'x0')
        x0 = opts.x0;
    else
        x0 = zeros(size(b));
    end

    if isfield(opts,'m')
        m = opts.m;
    else
        m = 100;
    end

    if isfield(opts,'tol')
        tol = opts.tol;
    else
        tol = 1e-3;
    end

    if isfield(opts,'xtrue')
        xtrue = opts.xtrue;
    else
        xtrue = [];
    end
    isxtrue = ~isempty(xtrue);

    if isfield(opts,'isrr')
        isrr = opts.isrr;
    else
        isrr = true;
    end

    if isfield(opts,'checkTimeout')
        checkTimeout = opts.checkTimeout;
    else
        checkTimeout = true;
    end
    checkCounter = 0;


    if nargout == 2
        out = struct();
        out.X = [];
        out.R = [];
        out.resvec = [];
        out.approxresvec = [];
        out.errvec = [];
    elseif nargout > 2
        error('Too many outputs specified');
    else
        out = [];
    end
    
    isout = ~isempty(out);
    
    if ~isa(A,'function_handle')
        Aop = @(x) A*x;
    else
        Aop = A;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initialize data structures %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    s = zeros(m,1);
    c = zeros(m,1);
    [n,k] = size(U);
    r = b - Aop(x0);
    r0 = r;
    normr0 = norm(r0);
    if isout
        out.resvec = zeros(m+2,1);
        out.approxresvec = zeros(m+2,1);
        out.X = zeros(n,m+1);
        out.R = zeros(n,m+1);
        out.R = [out.R r];
        out.X = [out.X x0];
        out.resvec(1) = norm(r);
        out.approxresvec(1) = norm(r);
        if isxtrue
            out.errvec = zeros(m+2,3); 
            out.errvec(1,1 ) = norm(x0 - xtrue);  
            out.errvec(1,2) = norm(x0 - xtrue,1); 
            out.errvec(1,3) = norm(x0 - xtrue,inf);
        end
    end
    
    C = Aop(U); [C,R] = qr(C,0);
    
    normCoeff = norm(r - C*(C'*r))/norm(r);
    
    H = zeros(m+1,m);
    HH = zeros(m+1,m);
    Dj = zeros(m+1,k);
    Mj = Dj;
    V = zeros(n,m+1);
    rhs = zeros(m+1,1);
    VTr0 = zeros(m+1,1);
    
    if isrr
        V(:,1) = Aop(r); 
        beta = norm(V(:,1));
        V(:,1) = V(:,1)/beta;
        rhs(1) = V(:,1)' * r0;
        VTr0(1) = rhs(1);
    else
        V(:,1) = r; 
        beta = norm(V(:,1));
        V(:,1) = V(:,1)/beta;
        rhs(1) = beta;
    end
     
    Dj(1,:) = V(:,1)'*C;
    Mj(1,:) = Dj(1,:);
    
    s1 = U * ( R \ ( C' * r ) );
    
    for iter = 1:m
        %%%%%%%%%%%%%%%%
        % Arnoldi step %
        %%%%%%%%%%%%%%%%
        V(:,iter+1) = Aop(V(:,iter));
        for i=1:iter
            H(i,iter) = V(:,i)'*V(:,iter+1);
            V(:,iter+1) = V(:,iter+1) - H(i,iter)*V(:,i);
        end
        H(iter+1,iter) = norm(V(:,iter+1));
        V(:,iter+1) = V(:,iter+1)/H(iter+1,iter);
        HH(:,iter) = H(:,iter);
        
        Dj(iter+1,:) = V(:,iter+1)'*C;
        Mj(iter+1,:) = Dj(iter+1,:);
        
        if isrr
            rhs(iter+1) = V(:,iter+1)' * r0;
            VTr0(iter+1) = rhs(iter+1);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Perform plane rotations on new column %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i = 1:iter-1
            h1 = H(i,iter);
            h2 = H(i+1,iter);
            H(i,iter)   = conj(c(i)) * h1 + s(i) * h2;
            H(i+1,iter) = -s(i) * h1 + c(i) * h2;
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Calculate new plane rotation %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
        y1 = H(iter,iter);
        y2 = H(iter+1,iter);
        rh1 = rhs(iter);
        rh2 = rhs(iter+1);
        m1 = Mj(iter,:);
        m2 = Mj(iter+1,:);
        if y2 == 0.0
            c(iter) = y1/conj(y1);
            s(iter) = 0.0;
        elseif abs(y2) > abs(y1)
            h1 = (y1/y2);
            s(iter) = 1.0 / sqrt(1 + abs(h1) * abs(h1));
            c(iter) = s(iter) * h1;
        else
            h1 = (y2/abs(y1));
            c(iter) = y1 / abs(y1) / sqrt(1 + h1 * h1);
            s(iter) = h1 / sqrt(1 + h1 * h1);
        end
        h1       = y1;
        h2       = y2;
        H(iter,iter)   = conj(c(iter)) * h1 + s(iter) * h2;
        H(iter+1,iter) = 0;%-s(j) * h1 + c(j) * h2;
        rhs(iter)   = conj(c(iter)) * rh1 + s(iter) * rh2;      
        rhs(iter+1) = -s(iter) * rh1 + c(iter) * rh2;  
        Mj(iter,:) = conj(c(iter)) * m1 + s(iter) * m2;
        Mj(iter+1,:) = -s(iter) * m1 + c(iter) * m2;
        
        if isrr
            resapprox = sqrt(abs(rhs(iter+1))^2);
        else
            resapprox = abs(rhs(iter+1));
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % If user specified output structure    %
        % explicitely construct entire solution %
        % and residual history and resvecs      %
        % If xtrue specified, calculate errors  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if isout
            rhsSub = rhs(1:iter) - Mj(1:iter,:) * ( C' * r0 );
            ysub = H(1:iter,1:iter) \ ((eye(iter) - Mj(1:iter,:) * Mj(1:iter,:)') \ rhsSub);
            t = V(:,1:iter)*ysub(1:iter);
            s2 = -U * ( R \ ( Dj(1:iter+1,:)' * ( HH(1:iter+1,1:iter) * ysub(1:iter) ) ) );
            out.X(:,iter+1) = x0+ s2 + t + s1;
            out.R(:,iter+1) = b - Aop(out.X(:,iter+1));
            out.resvec(iter+1) = norm(out.R(:,iter+1));
            out.approxresvec(iter+1) = normCoeff * resapprox;
        
            if isxtrue
                out.errvec(iter+1,1) = norm(out.X(:,iter+1)-xtrue);
                out.errvec(iter+1,2) = norm(out.X(:,iter+1)-xtrue,1);
                out.errvec(iter+1,3) = norm(out.X(:,iter+1)-xtrue,inf);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%
        % Convergence check     %
        % If not yet converged  %
        % update scaling factor %
        % and keep going        %
        %%%%%%%%%%%%%%%%%%%%%%%%%
        if normCoeff * resapprox < normr0*tol && checkCounter == 0
            rhsSub = rhs(1:iter) - Mj(1:iter,:) * ( C' * r0 );
            ysub = H(1:iter,1:iter) \ ((eye(iter) - Mj(1:iter,:) * Mj(1:iter,:)') \ rhsSub);
            t = V(:,1:iter)*ysub(1:iter);
            s2 = -U * ( R \ ( Dj(1:iter+1,:)' * ( HH(1:iter+1,1:iter) * ysub(1:iter) ) ) );
            x = x0+ s2 + t + s1;
            r = b - Aop(x);
            if norm(r) < normr0*tol
                break;
            else
                normCoeff = norm(r - C*(C'*r))/norm(r);
                checkCounter = checkTimeout;
            end
        elseif checkCounter > 0
            checkCounter = checkCounter - 1;
        end
    end
    
    if iter >= m
        rhsSub = rhs(1:iter) - Mj(1:iter,:) * ( C' * r0 );
        ysub = H(1:iter,1:iter) \ ((eye(iter) - Mj(1:iter,:) * Mj(1:iter,:)') \ rhsSub);
        t = V(:,1:iter)*ysub(1:iter);
        s2 = -U * ( R \ ( Dj(1:iter+1,:)' * ( HH(1:iter+1,1:iter) * ysub(1:iter) ) ) );
        x = x0+ s2 + t + s1;
    end
    
    if isout
       out.resvec = out.resvec(1:iter+1);
       out.approxresvec = out.approxresvec(1:iter+1);
       out.X = out.X(:,1:iter+1);
       out.R = out.R(:,1:iter+1);
       if isxtrue
           out.errvec = out.errvec(1:iter+1,:);
       end
    end
end
