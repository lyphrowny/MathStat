function [betas, ws] = min(A, inf, sup)
    [m, n] = size(A);
    rad = (sup - inf) / 2;
    mid = (sup + inf) / 2;
    
    c = [zeros(n, 1); ones(m, 1)];
    A = [[A, -diag(rad)]; [A, diag(rad)]];
    b = [mid; mid];
    
    c_type = [char(zeros(m, 1) + double("U")); char(zeros(m, 1) + double("L"))];
    b_type = char(zeros(n + m, 1) + double("C"));
    
    lb = [-Inf(n, 1); zeros(m, 1)];
    ub = Inf(n + m, 1);
    
    x = glpk(c, A, b, lb, ub, c_type, b_type);
    betas = x(1:n);
    ws = x(n+1:end);
endfunction