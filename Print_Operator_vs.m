clear all;

% ---------- Common grid ----------
fine = 25;
XBOX = [-150 150];
YBOX = [-150 150];
[xx, yy] = meshgrid(linspace(XBOX(1), XBOX(2), fine+1), ...
                    linspace(YBOX(1), YBOX(2), fine+1));
xx_q = xx(:)'; 
yy_q = yy(:)';

% ---------- Compute prox fields on this grid ----------
print_1  = zeros(2, numel(xx_q));  % Prox 1-norm
print_2  = zeros(2, numel(xx_q));  % Prox 2-norm
print_22 = zeros(2, numel(xx_q));  % Prox 2-norm squared

for c = 1:numel(xx_q)
    q = [xx_q(c); yy_q(c)];
    print_1(:,c)  = J1(10, q);
    print_2(:,c)  = J_2norm(1, q);
    print_22(:,c) = q/2;
end

% ---------- Learned 1 (Butterfly) ----------
load('Learned_Operators/INCREASING_BFLY_Sigma10_2000pts.mat');
Points = T.X;
print_T1 = zeros(2, numel(xx_q));
for c = 1:numel(xx_q)
    q = [xx_q(c); yy_q(c)];
    print_T1(:,c) = Op_T(q, A, B, T, X, Zsol, Points);
end

% ---------- Learned 2 (MNIST) ----------
load('Learned_Operators/INCREASING_MNIST_Sigma10_2000pts.mat');
Points = T.X;
print_T2 = zeros(2, numel(xx_q));
for c = 1:numel(xx_q)
    q = [xx_q(c); yy_q(c)];
    print_T2(:,c) = Op_T(q, A, B, T, X, Zsol, Points);
end

% ---------- Save each as its own PDF ----------
save_quiver(xx_q, yy_q, print_T1,  XBOX, YBOX, 'Learned 1',          'looks_vs_B');
save_quiver(xx_q, yy_q, print_T2,  XBOX, YBOX, 'Learned 2',          'looks_vs_M');
save_quiver(xx_q, yy_q, print_1,   XBOX, YBOX, 'Prox 1-norm',        'looks_vs_1');
save_quiver(xx_q, yy_q, print_2,   XBOX, YBOX, 'Prox 2-norm',        'looks_vs_2');
save_quiver(xx_q, yy_q, print_22,  XBOX, YBOX, 'Prox 2-norm squared','looks_vs_22');

% ---------- Helper to plot & save ----------
function save_quiver(xxq, yyq, F, XBOX, YBOX, ttl, fname)
    figure('Color','w'); clf;
    % Scale arrows a bit so they are readable but not too dense:
    quiver(xxq, yyq, F(1,:)-xxq, F(2,:)-yyq, 0.9, 'Color',[0 0.4470 0.7410], 'LineWidth', 1);
    title(ttl);
    axis equal; xlim(XBOX); ylim(YBOX); box on;
    % If you want completely tight visuals, uncomment the next line:
    % axis off
    set(gcf,'PaperPosition',[0 0 15 15],'PaperSize',[15 15]);
    saveas(gcf, fname, 'pdf');
    close(gcf);
end