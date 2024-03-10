% Print the firmly nonexpansive operator T

fine = 20;
l = 1.5;
[xx,yy] = meshgrid([min(X(1,:))/l:(max(X(1,:))-min(X(1,:)))/(l*fine):max(X(1,:))/l],[min(X(2,:))/l:(max(X(2,:))-min(X(2,:)))/(l*fine):max(X(2,:))/l]);

%% Define Q
Points = T.X; Q = Polyhedron('V', Points(T.convexHull,:));

c=1;
clear print_T; clear print_1; clear print_2; clear print_22;
for i=1:fine+1
    for j=1:fine+1
        print_T(:,c) = Op_T([xx(j,i);yy(j,i)],A,B,T,X,Zsol,Q);
        print_1(:,c) = J1(10,[xx(j,i);yy(j,i)]);
        print_2(:,c) = J_2norm(1,[xx(j,i);yy(j,i)]);
        print_22(:,c) = [xx(j,i);yy(j,i)]/2;
        c = c+1;
        if mod(c,100) == 0
            c;
        end
    end
end

xx = reshape(xx,1,(fine+1)^2);
yy = reshape(yy,1,(fine+1)^2);

figure(11); % clf;
subplot(2,2,1); quiver(xx,yy,print_T(1,:)-xx,print_T(2,:)-yy,'Color',[0 0.4470 0.7410]); title('Learned'); hold on;
subplot(2,2,2); quiver(xx,yy,print_1(1,:)-xx,print_1(2,:)-yy,'Color',[0 0.4470 0.7410]); title('Prox 1-norm'); hold on;
subplot(2,2,3); quiver(xx,yy,print_2(1,:)-xx,print_2(2,:)-yy,'Color',[0 0.4470 0.7410]); title('Prox 2-norm'); hold on;
subplot(2,2,4); quiver(xx,yy,print_22(1,:)-xx,print_22(2,:)-yy,'Color',[0 0.4470 0.7410]); title('Prox 2-norm squared'); hold on;
set(gcf, 'PaperPosition', [0 0 30 15]); % Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [30 15]); % Set the paper to have width 5 and height 5.
saveas(gcf, 'looks_1', 'pdf') % Save figure