% Find the two closest points in X

x = rand(2,1);

XX = X;
k1 = dsearchn(XX', x');
XX(:,k) = 1000000*ones(d,1);
k2 = dsearchn(XX', x');

inv([XX(:,k1),XX(:,k2)])

