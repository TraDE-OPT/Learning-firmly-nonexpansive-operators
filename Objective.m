function [out,out1,out2] = Objective(Z,Zbar,T,A)

% Compute the objective function (we penalize being out of constraint)

J = size(T,1);

vect = [];
for j=1:J
    % vect(j) = norm(NEW_Lj(T,A(:,:,j),j,Z)*inv(A(:,:,j)));
    vect(j) = norm(NEW_Lj(T,A(:,:,j),j,Z));
end

out1 = max(vect); % Lipschitz constant
% out11 = mean(vect); % Mean of Lipschitz constants

out2 = 1/2*norm(Z-Zbar,'fro')^2/J;

out = out1+out2;

end

