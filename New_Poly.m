% DEMOPROJECTPOINT.M
% Demonstration of projecting a point y onto the convex hull of a set of points P in 2D.

% 1) Define some 2D points P
%    (Here we just generate random points for demonstration.)
P = [rand(10,1)*5, rand(10,1)*5];  % 10 random points in a 5x5 region
P = Points;

% 2) Define a query point y (pick something that might be outside the hull)
y = [2.5, 5.5];  % For instance, a point potentially above most of the data

% 3) Compute the closest point 'cp' in conv(P) to y
cp = projectPointOnto2DHull(P, y);

% 4) Plot everything
figure; hold on; grid on; axis equal;

% Plot the original points
plot(P(:,1), P(:,2), 'bo', 'MarkerFaceColor','b', 'DisplayName','Points P');

% Plot the query point y
plot(y(1), y(2), 'rx', 'MarkerSize',10, 'LineWidth',2, 'DisplayName','Query point y');

% Plot the convex hull (polygon) by connecting hull vertices
hullIdx = convhull(P(:,1), P(:,2));
plot(P(hullIdx,1), P(hullIdx,2), 'g-', 'LineWidth',2, 'DisplayName','Convex Hull');

% Plot the closest point cp
plot(cp(1), cp(2), 'ms', 'MarkerSize',8, 'LineWidth',2, 'DisplayName','Projection cp');

title('Demo: Project y onto the Convex Hull of P');
legend('Location','best');
hold off;



function closestPoint = projectPointOnto2DHull(P, y)
% PROJECTPOINTONTO2DHULL Returns the closest point in the 2D convex hull of P to y.
% 
%   INPUTS:
%     P  - n-by-2 array of points in R^2
%     y  - 1-by-2 query point [y_x, y_y]
%
%   OUTPUT:
%     closestPoint - 1-by-2 point on conv(P) that is closest to y
%
% Example usage:
%   P = [0 0; 1 0; 1 1; 0 1];  % a square
%   y = [0.5, 1.5];            % a point above the square
%   cp = projectPointOnto2DHull(P, y);

    % 1. Compute the convex hull indices
    hullIdx = convhull(P(:,1), P(:,2));
    hullPts = P(hullIdx, :);  % Hull vertices in order

    % 2. Check if y is inside the polygon formed by hullPts
    if inpolygon(y(1), y(2), hullPts(:,1), hullPts(:,2))
        % If y is inside, the projection is y itself
        closestPoint = y;
    else
        % If y is outside, project onto the hull boundary
        closestPoint = projectPointOntoPolygon(hullPts, y);
    end
end

% -------------------------------------------------------------------------
function cp = projectPointOntoPolygon(hullPoints, y)
% PROJECTPOINTONTOPOLYGON Returns the closest point from y to a polygon
% specified by 'hullPoints' in cyclic order. The polygon is assumed convex
% (in practice, hullPoints is the output of convhull).

    m = size(hullPoints, 1);
    bestDist = Inf;
    cp = [NaN, NaN];

    for i = 1:m
        % Current vertex is hullPoints(i,:)
        % Next vertex is hullPoints(i+1,:), wrapping around
        p1 = hullPoints(i, :);
        p2 = hullPoints(mod(i, m) + 1, :);

        % Compute the projection of y onto segment [p1, p2]
        [projPoint, dist] = pointSegmentDistance(p1, p2, y);

        % Update if this is a better distance
        if dist < bestDist
            bestDist = dist;
            cp = projPoint;
        end
    end
end

% -------------------------------------------------------------------------
function [closestPt, dist] = pointSegmentDistance(p1, p2, y)
% POINTSEGMENTDISTANCE Returns the closest point on the segment [p1,p2] to y
% and also the Euclidean distance.

    v = p2 - p1;    % Vector along the segment
    w = y - p1;     % Vector from p1 to y

    c1 = dot(w, v);
    c2 = dot(v, v);

    % Parameter t along the segment
    t = c1 / c2;

    if t < 0
        % Before p1
        closestPt = p1;
    elseif t > 1
        % Past p2
        closestPt = p2;
    else
        % Between p1 and p2
        closestPt = p1 + t * v;
    end

    dist = norm(y - closestPt);
end
