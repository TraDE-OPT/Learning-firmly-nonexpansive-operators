function Out = Projection(Q,x)

% Projection into the polygon Q 

v = Q.project(x);

Out = v.x+10^-4*(v.x-x); % Push a bit further into the Polygon, to be sure the point is inside

end