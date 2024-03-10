clear all;
rng(1)
tbxmanager restorepath
tol = 10^-6;
SIGMA = 10; % Level of noise for the dataset
N = 28; % image dimension for MINST dataset
% N = 540; % image dimension for BUTTERFLIES dataset
NN = N*N;

%% Create finite difference matrix
temp = speye(N)-[sparse(1,N);speye(N-1),sparse(N-1,1)]; temp = [temp(2:N,:);sparse(1,N)];
D1 = kron(speye(N),temp); D1 = D1(1:end-N,:);
D2 = speye(NN-N,NN)-[sparse(NN-N,N),speye(NN-N)]; D2 = -D2;
for i=1:N-1
    D2(i*N,:) = sparse(1,NN);
end
D = [D1;D2];
[d2,d1] = size(D);

fprintf('Creare Dataset...\n')
%% Create Dataset: MINST
XX = []; YYbar = [];
for i = 1:10
sample_image = imread([int2str(i-1),'.jpg']);
V = double(reshape(sample_image,NN,1)); % Reshaped image and make it double
temp = D*V; % This is the vector [Dx;Dy]
Ybar = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)];
V_noise = V + randn(NN,1)*SIGMA;
temp = D*V_noise; % This is the vector [Dx;Dy]
X = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)];
X = X'; Ybar = Ybar';
XX = [XX,X]; YYbar = [YYbar,Ybar];
end
X = XX; Ybar = YYbar;

% %% Create Dataset: BUTTERFLIES
% sample_image = imread('butterflies.jpg'); sample_image = rgb2gray(sample_image); 
% V = double(reshape(sample_image,NN,1)); % Reshaped image and make it double
% temp = D*V; % This is the vector [Dx;Dy]
% Ybar = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)];
% V_noise = V + randn(NN,1)*SIGMA;
% temp = D*V_noise; % This is the vector [Dx;Dy]
% X = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)];
% X = X'; Ybar = Ybar';
% XX = X; YYbar = Ybar;

%% Data reduction (we use a specific clustering here to reduce the number of points, one can use k-means or others)
fprintf('Data reduction...\n')
clusters = 250; % Decide the number of output data points
[X,Ybar] = Clustering(X',Ybar',clusters);

%% Symmetrize the data (we symmetrize the data in order to obtain a rotation invariant operator)
clear XX; clear YYbar;
ts = size(X,2);
XX(:,1:ts) = X; XX(:,ts+1:2*ts) = [X(1,:);-X(2,:)]; XX(:,2*ts+1:3*ts) = [-X(1,:);X(2,:)]; XX(:,3*ts+1:4*ts) = -X;
YYbar(:,1:ts) = Ybar; YYbar(:,ts+1:2*ts) = [Ybar(1,:);-Ybar(2,:)]; YYbar(:,2*ts+1:3*ts) = [-Ybar(1,:);Ybar(2,:)]; YYbar(:,3*ts+1:4*ts) = -Ybar;
X = XX; Ybar = YYbar;
clear XX; clear YYbar;

%% We check for copies of points in the dataset and remove them
[X,I] = unique(X','rows'); X = X'; Ybar = Ybar(:,I);
d = 2; n = size(X,2);


Zbar = 2*(Ybar-1/2*X); % We search for firmly nonexpansive operators

%% Triangulation
T = DelaunayTri(X');
J = size(T,1);
A = zeros(d,d,J);
for j=1:J
    M = [];
    for i=T(j,2:end)
        M = [M,X(:,i)-X(:,T(j,1))];
    end
    A(:,:,j) = M;
end

%% Finding solution with ADMM
lip_constant = 1; % Here we decide the Lipschitz constant of the target operator. It can be any positive number. for our application it has to be less or equal than 1.
fprintf('Finding solution with ADMM...\n')
maxit = 10000; submaxit = 100; % Maximum number of iterations for outer and inner loops
tol = J*10^-10; subtol = 10^-1; % Tollerance for outer and inner loops
% We need the norm of the linear operator L. First we compute the norm of all Lj
for j = 1:J
    norms(j) = norm(inv(A(:,:,j)));
end
norm_L = sqrt(8)*max(norms); % Norm of L (estimate)
Z = zeros(d,n); U = zeros(d,d,J); lambda = U; % Inizialize 
err = 2*tol; res = []; fk = []; k = 0; not_lip = 1;
total_printed = maxit/100;
while k<maxit && err>tol
    rho = min(0.01*k/1000,1); % MNIST. To guarantee convergence it stabilizes after some iterations
    % rho = min((k^2+1)/(k+1000),1000); % BUTTERFLIES. To guarantee convergence it stabilizes after some iterations

    sigma = 1/(1+rho*norm_L^2); % To guarantee convergence of gradient descent
    i = 0; suberr = 2*subtol;
    while i<min(k,submaxit) && suberr>subtol
        Zold = Z;
        Z = Z - sigma*( Z-Zbar + NEW_L_t(T,A,J,n,d, rho*(NEW_L(T,A,J,d,Z)-U)+lambda) );
        suberr = norm(Z-Zold,'fro');
        i = i+1;
    end
    suberr1 = suberr;

    sigma = 1/rho; % To guarantee convergence of projected gradient method
    i = 0; suberr = 2*subtol;
    while i < min(k,submaxit) && suberr > subtol
        Uold = U;
        U = Proj_C(lip_constant,J,d, U + sigma*( rho*(NEW_L(T,A,J,d,Z)-U) + lambda ) );
        suberr = norm(U-Uold,'fro');
        i = i+1;
    end
    suberr2 = suberr;

    lambda_old = lambda;
    lambda = lambda_old+rho*(NEW_L(T,A,J,d,Z)-U);

    k = k+1; err = norm(lambda_old-lambda,'fro')+suberr1+suberr2; res(k) = err; [fk(k),out1(k),out2(k)] = Objective_function(Z,Zbar,T,A);
    not_lip = out1(k);
    if mod(k,maxit/total_printed) == 0
        fprintf('Iteration %d: residual = %f, Objective = %f\n',k,res(k),fk(k))
        Zsol = Z;
        Script_print_solutions
        save(['Images\FINAL_1_Sigma',int2str(SIGMA),'_',int2str(clusters*4),'pts'],'A','B','T','X','Zsol','Zbar','J');
    end
end

figure(1); clf; semilogy(out1); hold on; title('Non-Lipschitz');
figure(2); clf; semilogy(res); hold on;  title('Residuals');

Zsol = Z;
Ybar = 1/2*X+1/2*Zbar; 
Ysol = 1/2*X+1/2*Zsol; % We search for firmly nonexpansive operators

figure(3); clf;
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
figure(4); clf;
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


%save(['Images\FINAL_MNIST_train_Sigma',int2str(SIGMA),'_',int2str(clusters*4),'pts'],'A','B','T','X','Zsol','Zbar','J','SIGMA')
% save(['Operators\FINAL_ButterfliesTrain_Sigma',int2str(SIGMA),'_',int2str(clusters*4),'pts'],'A','B','T','X','Zsol','Zbar','J','SIGMA')