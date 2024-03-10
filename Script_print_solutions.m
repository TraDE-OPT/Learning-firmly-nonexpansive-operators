LW = 3; font = 20; % Linewidth and fontsize

figure(1); clf; semilogy(out1,'LineWidth',LW); hold on; title('Non-Lipschitz'); set(gca,'fontsize',font)
figure(2); clf; semilogy(res,'LineWidth',LW); hold on;  title('Residuals'); set(gca,'fontsize',font)

Zsol = Z;
Ysol = 1/2*X + 1/2*Zsol; % We search for firmly nonexpansive operators

figure(3); clf;
subplot(3,1,1); scatter(X(1,:),X(2,:),'k'); hold on;
subplot(3,1,2); scatter(Ybar(1,:),Ybar(2,:),'b'); hold on;
subplot(3,1,3); scatter(Ysol(1,:),Ysol(2,:),'r'); hold on;

B = zeros(d,d,J);
for j=1:J
    M = [];
    for i=T(j,2:end)
        M = [M,Zsol(:,i)-Zsol(:,T(j,1))];
    end
    B(:,:,j) = M;
end

figure(4); clf;
subplot(2,1,1); quiver(X(1,:),X(2,:),Ybar(1,:)-X(1,:),Ybar(2,:)-X(2,:)); hold on;
subplot(2,1,2); quiver(X(1,:),X(2,:),Ysol(1,:)-X(1,:),Ysol(2,:)-X(2,:)); hold on;

[out,out1_s,out2_s] = Objective(Zsol,Zbar,T,A);
fprintf('Out of constraint: %d, Loss: %d\n',out1_s,out2_s);

norms_of_jac = zeros(J,1);
for j=1:J
    norms_of_jac(j) = norm(B(:,:,j)*inv(A(:,:,j)));
end
figure(10); clf; scatter(1:J,norms_of_jac,'LineWidth',LW); hold on; plot(ones(J,1),'LineWidth',LW); set(gca,'fontsize',font)
set(gcf, 'PaperPosition', [0 0 30 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [30 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'lip_const_1', 'pdf') %Save figure

Print_Operator