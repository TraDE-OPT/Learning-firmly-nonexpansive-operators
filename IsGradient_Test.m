%% === Is the learned operator a gradient? ===
clear all;

% Style
LW   = 3;
font = 20;

% Load
load('Learned_Operators/INCREASING_MNIST_Sigma10_2000pts.mat'); % A,B,T,X,Zsol
% load('Learned_Operators/INCREASING_BFLY_Sigma10_2000pts.mat'); % A,B,T,X,Zsol
Points = T.X;

% --- Circle-test options
opts = struct;
opts.radii        = [10 20 40 60 80 100];
opts.numCenters   = 25;
opts.N            = 2048;
opts.seed         = 1;
opts.showProgress = false;

% --- Operators
R90       = [0 -1; 1 0];
T_learned = @(x) Op_T(x, A, B, T, X, Zsol, Points);

%% -------------------- Learned operator --------------------
resL  = check_T_is_gradient_via_circles_Tfun(T_learned, opts);
NC_L  = resL.table.NormCirc;
meanL = mean(NC_L);

fprintf('Mean NormCirc (learned): %.6e\n', meanL);

%% -------------------- α-grid sweep ------------------------
% Dense α grid (includes 0 and many tiny values up to 1)
alphas_grid = [0, logspace(-6, 0, 24)];   % 0, 1e-6, ..., 1
mean_alpha  = zeros(size(alphas_grid));

for i = 1:numel(alphas_grid)
    a    = alphas_grid(i);
    Tfun = @(x) a*(R90*x) + (1-a)*x;
    resA = check_T_is_gradient_via_circles_Tfun(Tfun, opts);
    mean_alpha(i) = mean(resA.table.NormCirc);
end

% Exact alpha* on this grid: nearest mean to the learned mean
[~,idxStar] = min(abs(mean_alpha - meanL));
alpha_star  = alphas_grid(idxStar);
mean_at_star = mean_alpha(idxStar);

fprintf('alpha* (grid):         %.6g\n', alpha_star);
fprintf('mean at alpha*:        %.6e (diff to learned = %.2e)\n', ...
        mean_at_star, abs(mean_at_star - meanL));

%% -------------------- Plot 1: mean vs alpha (semilogy) -----------------
hfig1 = figure(6); clf;
h1 = loglog(alphas_grid, mean_alpha, '-o', 'LineWidth', LW, 'MarkerSize', 6, ...
              'DisplayName', 'alpha -> mean(NormCirc)'); hold on;
h2 = yline(meanL, '--', 'LineWidth', LW, 'DisplayName', 'learned mean');
h3 = xline(alpha_star, ':', 'LineWidth', LW, 'DisplayName', 'alpha*');
loglog(alpha_star, mean_at_star, 'kp', 'MarkerFaceColor','k', 'MarkerSize',10, ...
         'DisplayName', 'match');

xlabel('\alpha  (T_\alpha = \alpha R_{90}x + (1-\alpha)x)');
ylabel('Mean NormCirc');
title('Mean normalized circulation vs \alpha');
legend([h1 h2 h3], 'Location','best', 'Interpreter','none');
set(gca,'FontSize',font);
% grid on; box off;

% Save as PDF
set(hfig1, 'PaperPosition', [0 0 30 15]); % Position plot at left hand corner with width 30 and height 15.
set(hfig1, 'PaperSize', [30 15]); % Set the paper to have width 30 and height 15.
saveas(hfig1,'IsitGradient_MNIST_alpha','pdf');
% saveas(hfig1,'IsitGradient_BFLY_alpha','pdf');

%% -------- Plot 2: learned deviations from its mean (histogram) ---------
hfig2 = figure(7); clf;
dev = NC_L;   % deviation from the learned mean on each circle
histogram(dev, 'NumBins', 30, 'Normalization', 'pdf', ...
          'EdgeColor', 'k', 'LineWidth', LW, 'FaceAlpha', 0.4, ...
          'DisplayName', 'deviation'); hold on;
xline(meanL,'--','LineWidth',LW,'DisplayName','mean');

xlabel('NormCirc');
ylabel('Density');
title('Learned operator: deviation from mean NormCirc');
legend('Location','best', 'Interpreter','none');
set(gca,'FontSize',font);
% grid on; box off;

% Save as PDF
set(hfig2, 'PaperPosition', [0 0 30 15]); % Position plot at left hand corner with width 30 and height 15.
set(hfig2, 'PaperSize', [30 15]); % Set the paper to have width 30 and height 15.
saveas(hfig2,'IsitGradient_MNIST_stat','pdf');
% saveas(hfig2,'IsitGradient_BFLY_stat','pdf');

%% ---------------- Helper: circle test for a given T(x) -----------------
function results = check_T_is_gradient_via_circles_Tfun(Tfun, opts)
    if nargin < 2, opts = struct(); end
    if ~isfield(opts,'radii'),        opts.radii = [10 20 40 60 80 100]; end
    if ~isfield(opts,'numCenters'),   opts.numCenters = 25; end
    if ~isfield(opts,'N'),            opts.N = 2048; end
    if ~isfield(opts,'domainSquare'), opts.domainSquare = [-150 150; -150 150]; end
    if ~isfield(opts,'seed'),         opts.seed = 1; end
    if ~isfield(opts,'showProgress'), opts.showProgress = false; end

    rng(opts.seed);
    N = opts.N;
    thetas = (0:N-1) * (2*pi/N);
    sinth  = sin(thetas); 
    costh  = cos(thetas);
    dtheta = 2*pi/N;

    bx = opts.domainSquare(1,:); 
    by = opts.domainSquare(2,:);

    entries = struct([]); idx = 0;

    for r = opts.radii
        xmin = bx(1) + r; xmax = bx(2) - r;
        ymin = by(1) + r; ymax = by(2) - r;
        Cx = xmin + (xmax - xmin).*rand(opts.numCenters,1);
        Cy = ymin + (ymax - ymin).*rand(opts.numCenters,1);

        for c = 1:opts.numCenters
            idx = idx + 1;
            center = [Cx(c); Cy(c)];

            pts     = center + r .* [costh; sinth];   % 2 x N
            dpos_dt = r .* [-sinth; costh];          % 2 x N

            Tvals = zeros(2,N);
            for i = 1:N
                Tvals(:,i) = Tfun(pts(:,i));
            end

            integrand    = sum(Tvals .* dpos_dt, 1);
            circIntegral = dtheta * sum(integrand);

            speed     = sqrt(sum(Tvals.^2,1));
            meanSpeed = mean(speed);
            circLen   = 2*pi*r;
            NormCirc  = abs(circIntegral) / (meanSpeed*circLen + eps);

            entries(idx).center   = center;
            entries(idx).R        = r;
            entries(idx).Integral = circIntegral;
            entries(idx).NormCirc = NormCirc;
        end
    end

    Cx = arrayfun(@(e)e.center(1), entries).';
    Cy = arrayfun(@(e)e.center(2), entries).';
    R  = arrayfun(@(e)e.R,        entries).';
    I  = arrayfun(@(e)e.Integral, entries).';
    NC = arrayfun(@(e)e.NormCirc, entries).';

    tbl = table(R, Cx, Cy, I, NC, 'VariableNames',{'R','Cx','Cy','Integral','NormCirc'});
    results.entries = entries;
    results.table   = tbl;
    results.opts    = opts;
end
