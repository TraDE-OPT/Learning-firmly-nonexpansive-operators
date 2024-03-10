function [OUT] = Proj_C(par,J,d,UU)

% Projection into the set of matrices with singular values lower than "par"

    OUT = zeros(d,d,J);
    for j = 1:J
        [U,S,V] = svd(UU(:,:,j));
        S = min(S,par);
        OUT(:,:,j) = U*S*V';
    end

end

