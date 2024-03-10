function [y] = J1(par,x)

% Compute the prox of the 1-norm

y = max(abs(x)-par,0).*sign(x);

end

