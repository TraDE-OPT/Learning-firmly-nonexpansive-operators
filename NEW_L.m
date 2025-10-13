% function [OUT] = NEW_L(T,A,J,d,Z)
% % Compute the full linear operator L
% OUT = zeros(d,d,J);
% for j=1:J
%     OUT(:,:,j) = NEW_Lj(T,A(:,:,j),j,Z);
% end
% end

function OUT = NEW_L(T, A, J, d, Z)
% L : R^{d x n} -> R^{d x d x J}
% (Z_cols - Z_base) / A_j
OUT = zeros(d,d,J);
for j = 1:J
    base  = Z(:, T(j,1));
    cols  = T(j,2:end);
    diffs = Z(:, cols) - base;          % d x d
    OUT(:,:,j) = diffs / A(:,:,j);      % right-multiply by inv(A_j)
end
end

