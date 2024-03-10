function [out,out1,out2] = Objective_function(Z,Zbar,T,A)

% Compute the objective function (we penalize being out of constraint)

J = size(T,1);
d = size(A,1);

out1 = 0;
U = NEW_L(T,A,J,d,Z);
for j=1:J
    out1 = out1 + max(norm(U(:,:,j))-1,0);
end

out2 = 1/2*norm(Z-Zbar,'fro')^2;

out = out1+out2;

end

