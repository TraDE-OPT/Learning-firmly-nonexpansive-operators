function y = J_2norm(par, x)

% Compute the prox of par*||x||_2

n = norm(x);
if n <= eps
    y = zeros(size(x));
else
    scale = max(1 - par/n, 0);
    y = scale * x;
end
end