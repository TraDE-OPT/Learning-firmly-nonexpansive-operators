% function [OUT] = NEW_L_t(T,A,J,n,d,U)
% 
% % Compute the transpose of the full linear operator L
% 
% OUT = zeros(d,n);
% 
% for j=1:J
% 
% OUT = OUT + NEW_Lj_t(T,A(:,:,j),j,U(:,:,j),n,d);
% 
% end
% 
% end

function OUT = NEW_L_t(T, A, J, n, d, U)
% L^T : R^{d x d x J} -> R^{d x n}
% For each j: V_j = U_j / A_j' ; scatter V_j to cols and -sum(V_j) to base
OUT = zeros(d, n);
for j = 1:J
    base = T(j,1);
    cols = T(j,2:end);
    V = U(:,:,j) / (A(:,:,j)');         % right-multiply by inv(A_j') FIRST
    OUT(:, cols) = OUT(:, cols) + V;     % add to the "other" vertices
    OUT(:, base) = OUT(:, base) - sum(V, 2);  % subtract sum to the base vertex
end
end