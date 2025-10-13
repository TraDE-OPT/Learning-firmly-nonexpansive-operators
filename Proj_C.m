function [OUT] = Proj_C(par,J,d,UU)

% Projection into the set of matrices with singular values lower than "par"

    OUT = zeros(d,d,J);
    for j = 1:J
        [U,S,V] = svd(UU(:,:,j));
        S = min(S,par);
        OUT(:,:,j) = U*S*V';
    end

end

% % TO COMPUTE THE 2x2 CASE THIS SHOULD BE FASTER
% function OUT = Proj_C(par, J, d, UU)
% % Project 2x2 matrices UU(:,:,j) onto singular values <= par
% % Extract entries
% a = squeeze(UU(1,1,:));
% b = squeeze(UU(1,2,:));
% c = squeeze(UU(2,1,:));
% d = squeeze(UU(2,2,:));
% 
% % Compute largest singular value
% trS2 = a.^2 + b.^2 + c.^2 + d.^2;
% detS2 = (a.*d - b.*c).^2;
% sigma_max = sqrt(0.5*(trS2 + sqrt(trS2.^2 - 4*detS2)));
% 
% % Compute scaling factor
% scale = min(1, par ./ sigma_max);
% 
% % Apply scaling
% OUT = UU .* reshape(scale, [1 1 J]);
% end