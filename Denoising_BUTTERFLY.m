clear all
rng(1)
LW = 3; font = 20; % Linewidth and fontsize

%% Load
% tbxmanager restorepath
load('Learned_Operators\INCREASING_BFLY_Sigma10_2000pts.mat')

% Define
Points = T.X;
clear input
input_temp = imread('Butterfly.jpeg'); input_temp=rgb2gray(input_temp); 

figure(5); clf;
max_iim = 3; % Between 1 and 3
k1 = [300,350,450]; k2 = [100,250,400];

for iim = 1:3

% delta = 10; s = 0.6; l1 = 7.2; l2 = 0.28;
% delta = 20; s = 1.2; l1 = 14.8; l2 = 0.7;
delta = 30; s = 1.9; l1 = 20.8; l2 = 1.55;
fprintf('Section %d: \ndelta = %f, l1 = %f, l2 = %f, s = %f\n', iim, delta, l1, l2, s);

  
% k1 = randi(size(input_temp,1)-129); k2 = randi(size(input_temp,1)-129);
input_iim(iim,:,:) = double(input_temp(k1(iim)+1:k1(iim)+128,k2(iim)+1:k2(iim)+128));
input = zeros(128);
input(:,:) = double(input_iim(iim,:,:));

N = size(input,1); NN = N^2;
real_input = reshape(input,NN,1); % Clean image

input=real_input+delta*randn(NN,1); input=max(min(input,255),0); % Noisy image

%% Construct the finite difference operator
temp = speye(N)-[sparse(1,N);speye(N-1),sparse(N-1,1)]; temp=[temp(2:N,:);sparse(1,N)];
D1 = kron(speye(N),temp); D1=D1(1:end-N,:);
D2 = speye(NN-N,NN)-[sparse(NN-N,N),speye(NN-N)]; D2 = -D2;
for i=1:N-1
    D2(i*N,:) = sparse(1,NN);
end
D = [D1;D2];
[d2,d1] = size(D);

%% CP
maxit = 1000; tol = 10^-3;
for_conv = 1; % Parameter that can be adjusted to improve convergence speed
s = s*for_conv;
t = 1/(s*8);
tt = t*for_conv; % How much does f weight? Higher, higher the weight. In this way we can change t and s independently
x = zeros(size(D,2),1); y=zeros(size(D,1),1);
clear xx tempy;
c = 1; k = 0; err = 2*tol;
tic;
while k<maxit && tol<err
    xold = x;
    tempx = x-t*D'*y;
    x = (tempx+tt*input)/(1+tt); % Prox of square loss
    temp = y+s*D*(2*x-xold); % Is a vector with same dimension of (Dx,Dy)
    tempo = [temp(1:length(temp)/2),temp(length(temp)/2+1:end)]; %Is a matrix of the same dimension of Gradients matrix
    % temp(kk,:) is a vector in R2
    yold = y;
    parfor kk = 1:size(tempo,1) %in order to use parfor you have to use the command "parpool("threads");" first
        tempy(:,kk) = tempo(kk,:)'-s*Op_T(tempo(kk,:)'/s,A,B,T,X,Zsol,Points); %this is the prox of gi^*, which is id - prox_gi
    end
    y1 = tempy(1,:)'; y2=tempy(2,:)';
    y = [y1;y2];
    
    k = k+1;
    residualx(k) = norm((xold-x)/t-D'*(yold-y));
    residualy(k) = norm((yold-y)/s-D*(xold-x));
    err = residualx(k) + residualy(k);
    if mod(20*k,maxit) == 0
        fprintf('Iteration: %d\n', k);
        xx(:,c) = x;
        c = c+1;
        figure(6); clf; semilogy(residualx(1:k),'LineWidth',LW); hold on; semilogy(residualy(1:k),'LineWidth',LW); title('Residuals'); legend('Primal','Dual'); set(gca,'fontsize',font)
        set(gcf, 'PaperPosition', [0 0 30 15]); %Position plot at left hand corner with width 5 and height 5.
        set(gcf, 'PaperSize', [30 15]); %Set the paper to have width 5 and height 5.
        saveas(gcf, 'convergence_PnPCP', 'pdf') %Save figure
    end
end
xsol = uint8(x); k_max=k;
toc

%% CP for H1-loss
s = 1; t = 1/(s*8);
x = zeros(size(D,2),1); y=zeros(size(D,1),1);
residuals = zeros(maxit,1);
k = 0; err = 2*tol;
while k<maxit && tol<err
    xold = x;
    tempx = x-t*D'*y;
    x = (tempx+t*input)/(1+t); %Prox of square loss
    temp = y+s*D*(2*x-xold);
    yold = y; 
    y = (l2/(l2+s))*temp; %Prox dual of the 2norm square (the higher the parameter the higher the contribution)
    k = k+1;
    err = norm(x-xold)+norm(y-yold);
end
x2norm = uint8(x);

%% CP for iso TV
s = 1; t = 1/(s*8);
x = zeros(size(D,2),1); y=zeros(size(D,1),1);
residuals = zeros(maxit,1);
k = 0; err = 2*tol;
while k < maxit && tol < err
    xold = x;
    tempx = x-t*D'*y;
    x = (tempx+t*input)/(1+t); %Prox of square loss
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

%% Plots
input = uint8(input); real_input = uint8(real_input);
ok = reshape(xsol,N,N);
figure(5); hold on; colormap('gray');
subplot(max_iim,5,(iim-1)*5+1); image(reshape(real_input,N,N)); hold on; title('True image')
subplot(max_iim,5,(iim-1)*5+2); image(reshape(input,N,N)); hold on; title('Noisy image')
subplot(max_iim,5,(iim-1)*5+3); image(reshape(x2norm,N,N)); hold on; title('H1')
subplot(max_iim,5,(iim-1)*5+4); image(double(reshape(x21norm,N,N))); hold on; title('TV')
subplot(max_iim,5,(iim-1)*5+5); image(ok); hold on; title('Learned');
set(gcf, 'PaperPosition', [0 0 30 15]); %Position plot at left hand corner with width 30 and height 15.
set(gcf, 'PaperSize', [30 15]); %Set the paper to have width 30 and height 15.
if delta == 10
    saveas(gcf, 'Butterflies_delta10', 'pdf') %Save figure
end
if delta == 20
    saveas(gcf, 'Butterflies_delta10', 'pdf') %Save figure
end
if delta == 30
    saveas(gcf, 'Butterflies_delta10', 'pdf') %Save figure
end

%% Print values for the table
fprintf('Ready to copy: MSE & $ %f $ & $ %f $ & $ %f $ & $ mathbf{%f} $ \n',norm(double(real_input-input)),norm(double(real_input-x2norm)),norm(double(real_input-x21norm)),norm(double(real_input-xsol)))
fprintf('PSNR (dB) & $ %f $ & $ %f $ & $ %f $ & $ mathbf{%f} $ \\ \n',psnr(input,real_input),psnr(x2norm,real_input),psnr(x21norm,real_input),psnr(xsol,real_input))
fprintf('SSIM & $ %f $ & $ %f $ & $ %f $ & $ mathbf{%f} $ \\ \n',ssim(input,real_input),ssim(x2norm,real_input),ssim(x21norm,real_input),ssim(xsol,real_input))
input=double(input); real_input=double(real_input);
end