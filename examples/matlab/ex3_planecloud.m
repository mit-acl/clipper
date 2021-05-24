%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLIPPER Example: Plane cloud registration
%
%   Uses planes extracted from two real LiDAR scans with <50% overlap.
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
%% Plane parameters segmented from two LiDAR scans

D1 = [
    0.99778409, -0.02919371, -0.05978833,   1.84071578;...
    0.00655776, -0.34994794,  0.93674619,   5.81443529;...
    0.03067185,  0.93082657,  0.36417186, -22.82330860;...
   -0.03095734,  0.91232313,  0.40829902, -24.11912204;...
]';

D2 = [
    -0.07169808126,  0.855164861,  0.513373592, -28.65209536;...
    0.99514624580,  0.078913239,  0.058793283, -21.00096958;
    -0.00156293830, -0.344498312,  0.938785636,   5.98810865;...
     0.08368147539, -0.930524190, -0.356541920,  29.41486128;...
]';

% ground truth associations
Agt = [1 4; 2 3; 3 2];

%% Plot input plane clouds

% colors to loop through
colors = [
    0.8500, 0.3250, 0.0980;...
    0.9290, 0.6940, 0.1250;...
    0.4940, 0.1840, 0.5560;...
    0.4660, 0.6740, 0.1880;...
    0.3010, 0.7450, 0.9330;...
    0.6350, 0.0780, 0.1840;
];

[X, Y] = meshgrid(-50:1:50);

figure(1), cla; grid on; hold on;
title('Planes from LiDAR 1');
axis equal; xlabel('X'); ylabel('Y'); zlabel('Z');
for i = 1:size(D1,2)
    P = D1(:,i);
    P = P/norm(P);
    Z = (P(1)*X + P(2)*Y + P(4)) / (-P(3) + 1e-12);
    surf(X,Y,Z,'LineStyle','none','FaceColor',colors(i,:),'FaceLighting','gouraud');
end
axis([-50 50 -50 50 -10 10])

figure(2), cla; grid on; hold on;
title('Planes from LiDAR 2');
axis equal; xlabel('X'); ylabel('Y'); zlabel('Z');
for i = 1:size(D2,2)
    P = D2(:,i);
    P = P/norm(P);
    Z = (P(1)*X + P(2)*Y + P(4)) / (-P(3) + 1e-12);
    surf(X,Y,Z,'LineStyle','none','FaceColor',colors(i,:),'FaceLighting','gouraud');
end
axis([-50 50 -50 50 -10 10])

%% Generate putative associations

% by passing in an empty initial association matrix, an all-to-all
% hypothesis will be formed, i.e., CLIPPER will consider if any point in
% D1 can be associated with any other point in D2 (in a one-to-one way).
A = [];

%% Run CLIPPER
params = struct;
params.sign = deg2rad(1.5);
params.epsn = 1;

% massage data into correct format
DD1 = [zeros(3,size(D1,2)); D1(1:3,:)];
DD2 = [zeros(3,size(D2,2)); D2(1:3,:)];
[M, C, A] = clipper_pointnormaldistance(DD1, DD2, A, params);

[u, idx, ~] = clipper(M, C);
Ain = A(idx,:)

%% check returned correspondences
[~,idxAgt] = sort(Agt(:,1));
[~,idxAin] = sort(Ain(:,1));
correct = norm(Ain(idxAin,:) - Agt(idxAgt,:))==0;
if ~correct
    disp(['Incorrect correspondence returned. Please check for '...
        'symmetries in input data or tune affinity scoring parameters.'])
end

%% Find optimal alignment given correspondences

AA = D1(:,Ain(:,1));
BB = D2(:,Ain(:,2));

N1 = AA(1:3,:);
N2 = BB(1:3,:);

[U,~,V] = svd(N2*N1');
Rhat = U*diag([1 1 det(U*V)])*V';

N = (Rhat*N1)';
that = (N'*N)\N'*(AA(4,:) - BB(4,:))';
% that = zeros(3,1);

That = [Rhat that; 0 0 0 1];

%% Plot aligned planes

[X, Y] = meshgrid(-50:1:50);

T_1_2 = inv(That);

PP = inv(T_1_2)'*D2;

figure(4), cla; grid on; hold on; title('registered');
axis equal; xlabel('X'); ylabel('Y'); zlabel('Z');
for i = 1:size(D1,2)
    P = D1(:,i);
    P = P/norm(P);
    Z = (P(1)*X + P(2)*Y + P(4)) / (-P(3) + 1e-12);
    surf(X,Y,Z,'LineStyle','none','FaceColor',colors(i,:),'FaceLighting','gouraud');
end
for i = 1:size(PP,2)
    P = PP(:,i);
    P = P/norm(P);
    Z = (P(1)*X + P(2)*Y + P(4)) / (-P(3) + 1e-12);
    surf(X,Y,Z,'LineStyle','none','FaceColor',colors(i,:),'FaceLighting','gouraud');
end
axis([-50 50 -50 50 -10 10])
