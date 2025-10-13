function [idx, C] = local_kmeans(X, K, maxIter, tol)
%LOCAL_KMEANS Minimal Lloyd's k-means with k-means++ init.
%   [IDX, C] = LOCAL_KMEANS(X, K) clusters the N-by-D matrix X
%   (rows = observations, columns = features) into K clusters.
%   IDX is N-by-1 of cluster assignments (1..K). C is K-by-D centroids.
%   Optional: maxIter (default 100), tol (default 1e-6).
%
%   This is a lightweight fallback for when the Statistics Toolbox license
%   is unavailable. It supports Euclidean distance only.

    if nargin < 3 || isempty(maxIter), maxIter = 100; end
    if nargin < 4 || isempty(tol),     tol     = 1e-6; end

    X = double(X);
    [n, d] = size(X);
    if K < 1 || K > n
        error('local_kmeans:K', 'K must be an integer in [1, size(X,1)].');
    end

    % --- k-means++ initialization
    C = zeros(K, d);
    idx = randi(n);               % pick first center uniformly
    C(1,:) = X(idx,:);
    % Squared distances to the nearest existing center
    D2 = sum((X - C(1,:)).^2, 2);

    for k = 2:K
        probs = D2 ./ sum(D2);
        edges = min(1, cumsum(probs));
        r = rand();
        idx = find(edges >= r, 1, 'first');
        C(k,:) = X(idx,:);
        D2 = min(D2, sum((X - C(k,:)).^2, 2));
    end

    prevC = C;
    idx = ones(n,1);  % preallocate

    for it = 1:maxIter
        % --- Assign step
        % compute squared distances to each centroid
        dist2 = zeros(n, K);
        for k = 1:K
            diff = X - C(k,:);
            dist2(:,k) = sum(diff.^2, 2);
        end
        [~, idx] = min(dist2, [], 2);  % 1..K

        % --- Update step
        emptyClusters = false(K,1);
        for k = 1:K
            members = (idx == k);
            if any(members)
                C(k,:) = mean(X(members, :), 1);
            else
                emptyClusters(k) = true;
            end
        end
        % Re-seed any empty clusters with a random data point
        if any(emptyClusters)
            reidx = randi(n, nnz(emptyClusters), 1);
            C(emptyClusters, :) = X(reidx, :);
        end

        % --- Convergence: max centroid shift
        if max(vecnorm(C - prevC, 2, 2)) < tol
            break
        end
        prevC = C;
    end
end