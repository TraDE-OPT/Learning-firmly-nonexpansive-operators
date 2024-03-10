function [y] = Op_T(x,A,B,T,X,Zsol,Q)

% x is a point in the plane in R^2
% A(:,:,j) and B(:,:,j) are Aj, Bj (A,B \in R^(d x d x J), where J number of triangles)
% T is the triangulation
% X data points in R^2n
% Zsol are the solution output relative to X

J = size(T,1);

j = pointLocation(T,x'); 
% If the point is not in the triangulation, project into it
while ~(j<=J)
    if randi(100) == 1
        fprintf('projecting...\n');
    end
    x = Projection(Q,x);
    j = pointLocation(T,x');
end

z = B(:,:,j)/A(:,:,j)*(x-X(:,T(j,1))) + Zsol(:,T(j,1));

y = 1/2*z + 1/2*x;

end

