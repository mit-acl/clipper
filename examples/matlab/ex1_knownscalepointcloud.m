%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLIPPER Example: Synthetic point cloud registration w/ known scale
%
%   Before running this example, use cmake to build the required mex fcns.
%   See README.md for more information.
% 
% For more details, please see the article
%   P.C. Lusk, K. Fathian, J.P. How, "CLIPPER: A Graph-Theoretic Framework
%       "for Robust Data Association," ICRA 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc;
addpath(genpath('build/bindings/matlab')) % for scoring invariants
addpath(genpath('matlab')) % for clipper algorithm
%% Generate model/target point cloud

% blue point cloud
D1 = [
      0 0 0;...
      2 0 0;...
      0 3 0;...
      2 2 0;...
    ]';

%% Generate data/source point cloud

% arbitrary transformation of data w.r.t model
th = pi/8;
R = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1];
t = [5 3 0]';

% apply inverse of transform to model cloud
D2 = R'*D1 - R'*t;

% decide how many points of D2 to remove from the end (sim partial scan)
m = 1;

% remove last point from D2 to create a partial scan
D2 = D2(:,1:(end-m));

%% Generate putative associations

% by passing in an empty initial association matrix, an all-to-all
% hypothesis will be formed, i.e., CLIPPER will consider if any point in
% D1 can be associated with any other point in D2 (in a one-to-one way).
A = [];

%% Run CLIPPER

params = struct;
params.sigma = 0.01;
params.epsilon = 0.06;

[M, C, A] = clipper_euclideandistance(D1, D2, A, params);

[u, idx, ~] = clipper(M, C);
Ain = A(idx,:);

%% Plot input point clouds

figure(1), cla; grid on, hold on;
title('Point Clouds', 'FontSize', 18);
scatter(D1(1,1:(end-m)), D1(2,1:(end-m)), 50, 'b', 'filled');
scatter(D1(1,(end-m):end), D1(2,(end-m):end), 50, 'b');
scatter(D2(1,:), D2(2,:), 50, 'r', 'filled');
axis equal;

%% Plot CLIPPER input

figure(2), cla; hold on;
title('CLIPPER Input', 'FontSize', 18);
for i = 1:size(A,1)
    
    if A(i,1) == A(i,2) % if a correct association (see assumptions)
        color = [0 1 0];
    else
        color = [1 0.5 1];
    end

    plot([D1(1,A(i,1)) D2(1,A(i,2))], [D1(2,A(i,1)) D2(2,A(i,2))],...
            'LineWidth', 1, 'Color', color);
    
    c = mean([D1(1:2,A(i,1)) D2(1:2,A(i,2))], 2);
    text(c(1), c(2), num2str(i),...
        'HorizontalAlignment', 'center', 'VerticalAlignment','middle');
end
scatter(D1(1,1:(end-m)), D1(2,1:(end-m)), 50, 'b', 'filled');
scatter(D1(1,(end-m):end), D1(2,(end-m):end), 50, 'b');
scatter(D2(1,:), D2(2,:), 50, 'r', 'filled');
axis equal;
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);

%% Plot CLIPPER output

figure(3), cla; hold on
title('CLIPPER Output', 'FontSize', 18);
for i = 1:size(Ain,1)
    
    if Ain(i,1) == Ain(i,2) % if a correct association (see assumptions)
        color = [0 1 0];
    else
        color = [1 0.5 1];
    end

    plot([D1(1,Ain(i,1)) D2(1,Ain(i,2))], [D1(2,Ain(i,1)) D2(2,Ain(i,2))],...
            'LineWidth', 1, 'Color', color);
end
scatter(D1(1,1:(end-m)), D1(2,1:(end-m)), 50, 'b', 'filled');
scatter(D1(1,(end-m):end), D1(2,(end-m):end), 50, 'b');
scatter(D2(1,:), D2(2,:), 50, 'r', 'filled');
axis equal;
set(gca,'YTickLabel',[]);
set(gca,'XTickLabel',[]);