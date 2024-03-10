function [OUT] = NEW_L(T,A,J,d,Z)

% Compute the full linear operator L

OUT = zeros(d,d,J);
for j=1:J
    OUT(:,:,j) = NEW_Lj(T,A(:,:,j),j,Z);
end

end

