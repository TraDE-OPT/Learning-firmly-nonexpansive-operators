function [XX,YY] = Data_reduction(X,Y,d_obj)

% Reduce the number of data points. The has a uniformly distributed distance to zero

d = size(X,2);

for i = 1:d
    C(i) = norm(X(:,i));
end

M = max(C);
c = 0;
for i = 1:d
    if norm(X(:,i)) >= c*M/d_obj
        c = c+1;
        XX(:,c) = X(:,i);
        YY(:,c) = Y(:,i);
    end
end


end

