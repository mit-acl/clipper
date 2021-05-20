%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CLIPPER Example: Data association of noisy Stanford Bunny with outliers
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

pcfile = 'examples/data/bun1k.ply'; % '../data/bun1k.py';
m = 1000;
n1 = 1000;
n2o = 250;
or = 0.95;
noisesig = 0.02;

% ground truth transformation
R_21 = axang2rotm([0 0 1 0.2]) ...
  * axang2rotm([0 1 0 0.6]) ...
  * axang2rotm([1 0 0 0]);
t_21 = [-2 -1 1]';
T_21 = [R_21 t_21; 0 0 0 1];

[D1, D2, Agt, A] = generateDataset(pcfile,m,n1,n2o,or,noisesig,T_21);

%% Run CLIPPER

params = struct;
params.sigma = 0.015;
params.epsilon = 0.02;

[M, C, A] = clipper_euclideandistance(D1, D2, A, params);

tic;
[u, idx, ~] = clipper(M, C);
t = toc;
Ain = A(idx,:);

% report precision and recall
[P,R] = scorePR(Ain,Agt);
fprintf(['CLIPPER classified %d of %d associations as inliers '...
            'with %.0f%% precision and %.0f%% recall in %.0f ms.\n'],...
            size(Ain,1),m,P*100,R*100, t*1e3);

%% Plot input point clouds with putative associations

figure(1), clf; grid on; hold on;
title('CLIPPER input');
scatter3(D1(1,:),D1(2,:),D1(3,:),30,'b','.');
scatter3(D2(1,1:n1),D2(2,1:n1),D2(3,1:n1),30,'r','.');
scatter3(D2(1,n1+1:end),D2(2,n1+1:end),D2(3,n1+1:end),10,'r');

for i = 1:size(A,1)
    if A(i,1) == A(i,2)
        color = [0 1 0];
        lw = 1.5;
    else
        color = [1 0.5 1];
        lw = 0.5;
    end
    plot3([D1(1,A(i,1)) D2(1,A(i,2))],...
          [D1(2,A(i,1)) D2(2,A(i,2))],...
          [D1(3,A(i,1)) D2(3,A(i,2))],...
            'LineWidth', lw, 'Color', color);
end
axis equal

%% Plot associations selected by CLIPPER

figure(2), clf; grid on; hold on;
title('CLIPPER output');
scatter3(D1(1,:),D1(2,:),D1(3,:),30,'b','.');
scatter3(D2(1,1:n1),D2(2,1:n1),D2(3,1:n1),30,'r','.');
scatter3(D2(1,n1+1:end),D2(2,n1+1:end),D2(3,n1+1:end),10,'r');

for i = 1:size(Ain,1)
    if Ain(i,1) == Ain(i,2)
        color = [0 1 0];
        lw = 1.5;
    else
        color = [1 0.5 1];
        lw = 0.5;
    end
    plot3([D1(1,Ain(i,1)) D2(1,Ain(i,2))],...
          [D1(2,Ain(i,1)) D2(2,Ain(i,2))],...
          [D1(3,Ain(i,1)) D2(3,Ain(i,2))],...
            'LineWidth', lw, 'Color', color);
end
axis equal

%% Helpers
function [precision, recall] = scorePR(A, Agt)
% SCOREPR Score precision-recall for estimated associations vs ground truth
    
    % total number of inliers, i.e., the relevant elements
    P = size(Agt,1); % P = TP + FN
        
    % total number of elements *selected* as inliers (but may not be)
    nS = size(A,1); % nS = TP + FP

    % Of the selected inliers, how many are true pos / false pos
    TP = 0; FP = 0;
    for i = 1:nS
        ivec = and(any(Agt(:,1)==A(i,1),2), any(Agt(:,2)==A(i,2),2));
        if any(ivec)
            % this association was a true positive
            TP = TP + 1;
        else
            % this association was a false positive
            FP = FP + 1;
        end
    end
    
    precision = TP / (TP + FP);
    recall = TP / P;
    
    if isnan(precision), warning('we have a NaN in precision!'), precision = 0; end
    if isnan(recall), warning('we have a NaN! in recall'), recall = 0; end
end

function [D1,D2,Agt,A] = generateDataset(pcfile, m, n1, n2o, or, sigma, T_21)
%GENERATE_DATASET Generates two point clouds with associations
%
%   inputs:
%       m       Number of total associations to generate (i.e., size of
%               affinity matrix)
%       n1      Number of points in view 1; sampled from pcfile
%       n2o     Number of additional points (outliers) in view 2; drawn
%               uniformly from a solid sphere of radius R.
%       or      Desired outlier ratio, i.e., wrong assoc. / m
%       sigma   Std dev of noise added to all points in view 2
%

% Clean samples of model point cloud
pcd = pcread(pcfile);

n2 = n1 + n2o;         % number of points in view 2
noa = round(m*or);     % number of outlier associations
nia = m - noa;         % number of inlier associations

if nia > n1
    error(['Cannot have more inlier associations than there are model '...
            'points. Increase outlier ratio or increase the number '...
            'of points to sample from original point cloud model.']);
end

% radius of outlier sphere
R = 1;

%% Select Points

% Downsample from original point cloud, sample randomly
I = randsample(size(pcd.Location,1),n1);
D1 = double(pcd.Location(I,:))';

% Rotate into view 2 using ground truth transformation
D2 = T_21(1:3,1:3)*D1 + T_21(1:3,4);

% Add noise uniformly sampled from a sigma cube around the true point
a = -[1;1;1]*sigma/2;
b =  [1;1;1]*sigma/2;
eta = a + (b-a).*rand(size(D2));

% Add noise to view 2
D2 = D2 + eta;

% Add outliers to view 2
O2 = randsphere(n2o,3,R)' + T_21(1:3,4) + mean(D1,2);
D2 = [D2 O2];

% Correct associations to draw from
Agood = repmat((1:n1)',1,2);

% Incorrect associations to draw from
Abad = zeros(n1*n2 - n1,2);
itr = 0;
for i = 1:n1
    for j = 1:n2
        if i == j, continue; end
        itr = itr + 1;
        Abad(itr,:) = [i,j];
    end
end

% Sample good and bad associations to satisfy total
% number of associations with the requested outlier ratio.
IAgood = randsample(size(Agood,1),nia);
IAbad = randsample(size(Abad,1),noa);
A = [Agood(IAgood,:); Abad(IAbad,:)];

% Ground truth associations
Agt = Agood(IAgood,:);

end

function X = randsphere(m,n,r)
 
% This function returns an m by n array, X, in which 
% each of the m rows has the n Cartesian coordinates 
% of a random point uniformly-distributed over the 
% interior of an n-dimensional hypersphere with 
% radius r and center at the origin.  The function 
% 'randn' is initially used to generate m sets of n 
% random variables with independent multivariate 
% normal distribution, with mean 0 and variance 1.
% Then the incomplete gamma function, 'gammainc', 
% is used to map these points radially to fit in the 
% hypersphere of finite radius r with a uniform % spatial distribution.
% Roger Stafford - 12/23/05
 
X = randn(m,n);
s2 = sum(X.^2,2);
X = X.*repmat(r*(gammainc(s2/2,n/2).^(1/n))./sqrt(s2),1,n);
end