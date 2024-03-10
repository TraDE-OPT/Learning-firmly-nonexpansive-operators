function [y] = J_2norm(par,x)

% Compute the prox of the 2-norm

y = max(norm(x)-par,0)*x/norm(x);

end

