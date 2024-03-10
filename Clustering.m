function [XX,YYbar] = Clustering(X,Ybar,K)

% Clustering using k-means

[~, C] = kmeans(X, K);

for i = 1:K
    [~,j] = min(pdist2(C(i,:),X));
    XX(i,:) = X(j,:); YYbar(i,:) = Ybar(j,:);
end

XX = XX'; YYbar = YYbar';

end

