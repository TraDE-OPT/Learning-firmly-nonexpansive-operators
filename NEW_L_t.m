function [OUT] = NEW_L_t(T,A,J,n,d,U)

% Compute the transpose of the full linear operator L

OUT = zeros(d,n);

for j=1:J

OUT = OUT + NEW_Lj_t(T,A(:,:,j),j,U(:,:,j),n,d);

end

end
