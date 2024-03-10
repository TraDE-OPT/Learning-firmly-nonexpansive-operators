function [OUT] = NEW_Lj_t(T,At,j,U,n,d)

% Compute the transpose of the linear operator L_j

U = U*inv(At)';

OUT = zeros(d,n);
c = 1;
for i = T(j,2:end)
    OUT(:,i) = U(:,c);
    OUT(:,T(j,1)) = OUT(:,T(j,1))-U(:,c);
    c = c+1;
end

end

