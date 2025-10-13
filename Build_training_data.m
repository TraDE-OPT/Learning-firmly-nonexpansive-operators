clear all;
rng(1)
tol = 10^-6;
SIGMA = 10; % Level of noise for the dataset
% N = 28; % image dimension for MINST dataset
N = 540; % image dimension for BUTTERFLIES dataset
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
% %% Create Dataset: MINST
% XX = []; YYbar = [];
% for i = 1:10
% sample_image = imread([int2str(i-1),'.jpg']);
% V = double(reshape(sample_image,NN,1)); % Reshaped image and make it double
% temp = D*V; % This is the vector [Dx;Dy]
% Ybar = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)];
% V_noise = V + randn(NN,1)*SIGMA;
% temp = D*V_noise; % This is the vector [Dx;Dy]
% X = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)];
% X = X'; Ybar = Ybar';
% XX = [XX,X]; YYbar = [YYbar,Ybar];
% end
% X = XX; Ybar = YYbar;

%% Create Dataset: BUTTERFLIES
sample_image = imread('butterflies.jpg'); sample_image = rgb2gray(sample_image); 
V = double(reshape(sample_image,NN,1)); % Reshaped image and make it double
temp = D*V; % This is the vector [Dx;Dy]
Ybar = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)];
V_noise = V + randn(NN,1)*SIGMA;
temp = D*V_noise; % This is the vector [Dx;Dy]
X = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)];
X = X'; Ybar = Ybar';
XX = X; YYbar = Ybar;

%% Data reduction (we use a specific clustering here to reduce the number of points, one can use k-means or others)
fprintf('Data reduction...\n')
clusters = 500; % Decide the number of output data points
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


Zbar = 2*Ybar-X; % We search for firmly nonexpansive operators
% Zbar = X - Ybar; % If we search for gradient step denoisers

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