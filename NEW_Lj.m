function [OUT] = NEW_Lj(T,At,j,Z)

% Compute the linear operator L_j

OUT = [];
for i = T(j,2:end)
    OUT = [OUT,Z(:,i)-Z(:,T(j,1))];
end    

OUT = OUT*inv(At);

end

