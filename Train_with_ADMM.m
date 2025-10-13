Build_training_data

%% Finding solution with ADMM
lip_constant = 0.99; % Here we decide the Lipschitz constant of the target operator. It can be any positive number. for our application it has to be less or equal than 1.
fprintf('Finding solution with ADMM...\n')
maxit = 200000; submaxit = 10; % Maximum number of iterations for outer and inner loops
tol = 10^-8; subtol = 10^-1; % Tollerance for outer and inner loops
% We need the norm of the linear operator L. First we compute the norm of all Lj
for j = 1:J
    norms(j) = norm(inv(A(:,:,j)));
end
norm_L = sqrt(8)*max(norms); % Norm of L (estimate)
Z = zeros(d,n); Z_old = Z; U = zeros(d,d,J); U_old = U; lambda = U; % Inizialize 
err = 2*tol; res = []; fk = []; k = 1; not_lip = 1;
total_printed = maxit/100; % print every 100 iterations
rho = 0.01;  % starting rho
tic;
while k<maxit

    rho = rho * (1+1/k^(1.1)); rho = min(rho, 100);

    sigma = 2/(2+rho*norm_L^2); % Best convergence of gradient descent for m-strongly convex: 2/(L+m)
    i = 0; suberr = 2*subtol;
    while i<submaxit && suberr>subtol
        Zold = Z;
        Z = Z - sigma*( Z-Zbar + NEW_L_t(T,A,J,n,d, rho*(NEW_L(T,A,J,d,Z)-U)+lambda) );
        suberr = norm(Z-Zold,'fro')/(eps+norm(Z,'fro'));
        i = i+1;
    end
    suberr1 = suberr;

    LZ = NEW_L(T,A,J,d,Z);  % compute once
    U_old = U;
    U = Proj_C(lip_constant,J,d, LZ + lambda/rho );
    suberr2 = norm(U-U_old,'fro')/(eps+norm(U,'fro'));

    lambda_old = lambda;
    lambda = lambda_old+rho*(LZ-U);

    r_norm = norm(LZ - U,'fro'); % primal residual
    s_norm = norm(rho*NEW_L_t(T,A,J,n,d, U - U_old),'fro'); % dual residual

    k = k+1; err = norm(lambda_old-lambda,'fro')/(eps+norm(lambda,'fro'))+suberr1+suberr2; res(k) = s_norm + r_norm; [fk(k),out1(k),out2(k)] = Objective(Z,Zbar,T,A);
    not_lip = out1(k);
    approx = out2(k);
    if mod(k,maxit/total_printed) == 0
        fprintf('Iteration %d: residual = %f, Objective = %f\n',k,res(k),fk(k))
        Zsol = Z;
        Script_print_solutions
        save(['Learned_Operators\INCREASING_BFLY_Sigma',int2str(SIGMA),'_',int2str(clusters*4),'pts'],'A','B','T','X','Zsol','Zbar','J');
        rho
        r_norm
        s_norm
    end
end
toc
figure(1); clf; semilogy(out1); hold on; title('Non-Lipschitz');
figure(2); clf; semilogy(out2); hold on;  title('Least squares');
figure(3); clf; semilogy(res); hold on;  title('Residuals');

Zsol = Z;
Ybar = 1/2*X+1/2*Zbar; Ysol = 1/2*X+1/2*Zsol; % We search for firmly nonexpansive operators
% Ybar = X - Zbar; Ysol = X - Zsol; % If we search for gradient step denoisers


figure(4); clf;
subplot(3,1,1); scatter(X(1,:),X(2,:),'k'); hold on;
subplot(3,1,2); scatter(Ybar(1,:),Ybar(2,:),'b'); hold on;
subplot(3,1,3); scatter(Ysol(1,:),Ysol(2,:),'r'); hold on;

%% Define B_j for all j = 1 , ... , J
B = zeros(d,d,J);
for j = 1:J
    M = [];
    for i = T(j,2:end)
        M = [M,Zsol(:,i)-Zsol(:,T(j,1))];
    end
    B(:,:,j) = M;
end

%% Plots
figure(5); clf;
subplot(2,1,1); quiver(X(1,:),X(2,:),Ybar(1,:)-X(1,:),Ybar(2,:)-X(2,:)); hold on;
subplot(2,1,2); quiver(X(1,:),X(2,:),Ysol(1,:)-X(1,:),Ysol(2,:)-X(2,:)); hold on;

[out,out1_s,out2_s] = Objective(Zsol,Zbar,T,A);
fprintf('Out of constraint: %d, Loss: %d\n',out1_s,out2_s);

norms_of_jac = zeros(J,1);
for j = 1:J
    norms_of_jac(j) = norm(B(:,:,j)*inv(A(:,:,j)));
end
figure(10); clf;  hold on; plot(norms_of_jac,'bo'); plot(ones(J,1),'LineWidth',3);
set(gcf, 'PaperPosition', [0 0 30 15]); % Position plot at left hand corner with width 30 and height 15.
set(gcf, 'PaperSize', [30 15]); % Set the paper to have width 30 and height 15.
saveas(gcf, 'test', 'pdf') % Save figure

Print_Operator
