%We have everything we need for computing T
clear all;
rng(1)
LW = 3; font = 20; % Linewidth and fontsize

%% Load
tbxmanager restorepath
load('Learned_Operators\FINAL_ButterfliesTrain_Sigma10_1000pts.mat')

%% Define Q
Points = T.X; Q = Polyhedron('V', Points(T.convexHull,:)); 

% input=imread('0.jpg');
% input=imread('starfish.png'); input=input(:,:,1);
% input=imread('starfish2.jpg'); input=input(:,:,1); input=input(1:490,61:550);
% input=imread('Butterfly.jpeg'); input=input(:,:,1);
% input=imread('Flower.jpeg'); input=input(:,:,1); input=input(501:800,501:800);
input = imread('montage.tif'); input = input(1:size(input,1)/2,size(input,1)/2+1:end); %Circles
% input=input(1:size(input,1)/2,1:size(input,1)/2); %Hello world
% input=input(size(input,1)/2+1:end,size(input,1)/2+1:end); %Parrot
% input=input(size(input,1)/2+1:end,1:size(input,1)/2); %Lena
% input=imread('balloons.png'); input=input(:,:,1);

input = double(input);

%input=imread('Butterfly_circles.jpeg'); input=rgb2gray(input);
%input=double(input);

N = size(input,1); NN = N^2;
real_input = reshape(input,NN,1); %Clean image
delta = 30;
input=real_input+delta*randn(NN,1); input=max(min(input,255),0); %Dirty image
%input = MRI image "dirty"
figure(5); clf; %% HERE! (clf;)%%
hold on; colormap('gray');
subplot(2,3,1); image(reshape(real_input,N,N)); hold on; title('Real image')
subplot(2,3,2); image(reshape(input,N,N)); hold on; title('Noisy image')

%% Construct the finite difference operator
temp = speye(N)-[sparse(1,N);speye(N-1),sparse(N-1,1)]; temp=[temp(2:N,:);sparse(1,N)];
D1 = kron(speye(N),temp); D1 = D1(1:end-N,:);
D2 = speye(NN-N,NN)-[sparse(NN-N,N),speye(NN-N)]; D2 = -D2;
for i=1:N-1
    D2(i*N,:) = sparse(1,NN);
end
D = [D1;D2];
[d2,d1] = size(D);

%% CP 1: Denoising using the learned prox with the Butterflies dataset
load Learned_Operators\FINAL_ButterfliesTrain_Sigma10_1000pts.mat
Points = T.X; Q = Polyhedron('V', Points(T.convexHull,:));
l1 = 25; l2 = 0.55; % delta = 30: l1 = 25; l2 = 0.55; delta = 20: l1 = 10; l2 = 0.4; delta = 10: l1 = 5; l2 = 0.3; 
maxit = 1000; tol = 10^-2;
% for_conv=1/100; % Parameter that can be adjusted to improve convergence speed (puts a constant in front of the problem...)
s = 4.5; %% delta = 30: s = 4.5; delta = 20: s = 3; delta = 10: s = 2;
for_conv=1; % Parameter that can be adjusted to improve convergence speed (puts a constant in front of the problem...)
s = s*for_conv; % the higher the s, the higher the parameter in front of the regularizer
t = 1/(s*8);
tt = t*for_conv; % How much does f weight? Higher, higher the weight. In this way we can change t and s independently
x = zeros(size(D,2),1); y = zeros(size(D,1),1);
clear xx tempy;
c = 1; k = 0; err = 2*tol;
tic;
while k < maxit && tol < err
    xold = x;
    tempx = x-t*D'*y;
    x = (tempx+tt*input)/(1+tt); % Prox of square loss
    temp = y+s*D*(2*x-xold); % Is a vector with same dimension of (Dx,Dy)
    tempo = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)]; %Is a matrix of the same dimension of Gradients matrix
    % temp(kk,:) is a vector in R2
    yold = y;
    for kk = 1:size(tempo,1) %in order to use parfor you have to use the command "parpool("threads");" first
        tempy(:,kk) = tempo(kk,:)'-s*Op_T(tempo(kk,:)'/s,A,B,T,X,Zsol,Q); %this is the prox of gi^*, which is id - prox_gi
    end
    y1 = tempy(1,:)'; y2 = tempy(2,:)';
    y = [y1;y2];
    
    k = k+1;
    residualx(k) = norm((xold-x)/t-D'*(yold-y));
    residualy(k) = norm((yold-y)/s-D*(xold-x));
    err = residualx(k)+residualy(k);
    if mod(100*k,maxit) == 0
        % fprintf('Iteration: %d\n', k);
        figure(5); hold on; subplot(2,3,3);
        image(reshape(x,N,N)); hold on; title('Reconstruction with learned prox');
        xx(:,c) = x;
        c = c+1;
        figure(6); clf; semilogy(residualx(1:k)); hold on; semilogy(residualy(1:k)); title('Residuals'); legend('Primal','Dual');
    end
end
xsol1 = uint8(x); k_max = k;
err
toc

%% CP 2: Denoising using the learned prox with the MNIST dataset
load Learned_Operators\FINAL_MNIST_train_Sigma10_1000pts.mat
Points = T.X; Q = Polyhedron('V', Points(T.convexHull,:));
maxit = 1000; tol = 10^-2;
% for_conv=1/100; % Parameter that can be adjusted to improve convergence speed (puts a constant in front of the problem...)
s = 7; %% delta = 30: s = 7; delta = 20: s = 6; delta = 10: s = 4;
for_conv = 1; % Parameter that can be adjusted to improve convergence speed (puts a constant in front of the problem...)
s = s*for_conv; % the higher the s, the higher the parameter in front of the regularizer
t = 1/(s*8);
tt = t*for_conv; % How much does f weight? Higher, higher the weight. In this way we can change t and s independently
x = zeros(size(D,2),1); y=zeros(size(D,1),1);
clear xx tempy;
c = 1; k = 0; err = 2*tol;
tic;
while k<maxit && tol<err
    xold=x;
    tempx=x-t*D'*y;
    x=(tempx+tt*input)/(1+tt); % Prox of square loss
    temp=y+s*D*(2*x-xold); % Is a vector with same dimension of (Dx,Dy)
    tempo=[temp(1:length(temp)/2),temp(length(temp)/2+1:end)]; %Is a matrix of the same dimension of Gradients matrix
    % temp(kk,:) is a vector in R2
    yold=y;
    for kk=1:size(tempo,1) % in order to use parfor you have to use the command "parpool("threads");" first
        tempy(:,kk)=tempo(kk,:)'-s*Op_T(tempo(kk,:)'/s,A,B,T,X,Zsol,Q); % this is the prox of gi^*, which is id - prox_gi
    end
    y1=tempy(1,:)'; y2=tempy(2,:)';
    y=[y1;y2];
    
    k=k+1;
    residualx(k)=norm((xold-x)/t-D'*(yold-y));
    residualy(k)=norm((yold-y)/s-D*(xold-x));
    err=residualx(k)+residualy(k);
    if mod(10*k,maxit)==0
        fprintf('Iteration: %d\n', k);
        figure(5); hold on; subplot(2,3,6);
        image(reshape(x,N,N)); hold on; title('Reconstruction with learned prox');
        xx(:,c)=x;
        c=c+1;
        figure(6); clf; semilogy(residualx(1:k)); hold on; semilogy(residualy(1:k)); title('Residuals'); legend('Primal','Dual');
    end
end
xsol2 = uint8(x); k_max = k;
err
toc

%% CP for iso TV
x = ones(size(D,2),1); y = ones(size(D,1),1);
residuals = zeros(maxit,1);
k = 0; err = 2*tol;
tic;
while k<maxit && tol<err
    xold = x;
    tempx = x-t*D'*y;
    x = (tempx+t*input)/(1+t); % Prox of square loss
    temp = y+s*D*(2*x-xold);
    tempo = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)];
    yold = y;
    for kk = 1:size(tempo,1)
        tempy(:,kk) = tempo(kk,:)'-s*J_2norm(l1/s,tempo(kk,:)'/s); %Test with TV (the higher the parameter the higher the contribution)
    end
    y1 = tempy(1,:)'; y2 = tempy(2,:)';
    y = [y1;y2];
    k = k+1;
    err = norm(x-xold)+norm(y-yold);
end
x21norm = uint8(x);
err
toc

%% CP for 2-norm
x = zeros(size(D,2),1); y = zeros(size(D,1),1);
residuals = zeros(maxit,1);
k = 0; err = 2*tol;
while k < maxit && tol < err
    xold = x;
    tempx = x-t*D'*y;
    x = (tempx+t*input)/(1+t); %Prox of square loss
    temp = y+s*D*(2*x-xold);
    yold = y; 
    y = l2*temp; %Prox dual of the 2norm square (the higher the parameter the higher the contribution) %% 0.6
    k = k+1;
    err = norm(x-xold)+norm(y-yold);
end
x2norm = uint8(x);
err

%% Plots
input = uint8(input); real_input = uint8(real_input);
ok1 = reshape(xsol1,N,N); ok2 = reshape(xsol2,N,N);
figure(100); clf; hold on; colormap('gray');
subplot(2,3,1); image(reshape(real_input,N,N)); hold on; title('Real image')
subplot(2,3,2); image(reshape(input,N,N)); hold on; title('Noisy image')
subplot(2,3,3); image(reshape(x2norm,N,N)); hold on; title('H1')
subplot(2,3,4); image(double(reshape(x21norm,N,N))); hold on; title('TV')
subplot(2,3,5); image(ok1); hold on; title('Learned 1 (butterfly)');
subplot(2,3,6); image(ok2); hold on; title('Learned 2 (MNIST)');
set(gcf, 'PaperPosition', [0 0 30 15]); %Position plot at left hand corner with width 30 and height 15.
set(gcf, 'PaperSize', [30 15]); %Set the paper to have width 30 and height 15.
saveas(gcf, 'Circles', 'pdf') %Save figure
figure(6); clf; semilogy(residualx(1:k_max)); hold on; semilogy(residualy(1:k_max)); title('Residuals'); legend('Primal','Dual');
set(gcf, 'PaperPosition', [0 0 30 15]); %Position plot at left hand corner with width 5 and height 5.
set(gcf, 'PaperSize', [30 15]); %Set the paper to have width 5 and height 5.
saveas(gcf, 'test', 'pdf') %Save figure

%% Print values for the table
fprintf('Ready to copy: MSE & $ %f $ & $ %f $ & $ %f $ & %f & $ mathbf{%f} $ \n',norm(double(real_input-input)),norm(double(real_input-x2norm)),norm(double(real_input-x21norm)),norm(double(real_input-xsol1)),norm(double(real_input-xsol2)))
fprintf('PSNR (dB) & $ %f $ & $ %f $ & $ %f $ & $ %f$ & $ mathbf{%f} $ \\ \n',psnr(input,real_input),psnr(x2norm,real_input),psnr(x21norm,real_input),psnr(xsol1,real_input),psnr(xsol2,real_input))
fprintf('SSIM & $ %f $ & $ %f $ & $ %f $ & $ %f $ & $ mathbf{%f} $ \\ \n',ssim(input,real_input),ssim(x2norm,real_input),ssim(x21norm,real_input),ssim(xsol1,real_input),ssim(xsol2,real_input))

input=double(input); real_input=double(real_input);