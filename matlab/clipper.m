%CLIPPER Search for the densest cluster given a graph's affinity matrix
%
% This code takes the weighted affinity matrix M of an undirected graph G 
% as input and outputs a dense cluster (fully-connected subgraph) of G.
% The densest (largest in size and weight) is sought for using projected
% gradient ascent. Any returned solution is guaranteed to be at least a
% dense cluster (i.e., a maximal clique in the case of a binary graph).
%
% Inputs:
%
%       M:  an nxn weighted, symmetric affinity matrix. In the case that M
%           is a binary matrix and C=M, then a maximal clique is returned.
%       C:  an nxn binary, symmetric matrix of hard constraints.
%           Zero entries of C are active constraints, which prevent the 
%           corresponding edge to be selected in the output.
%
% Outputs:
%
%       u:      final iterate of the gradient ascent scheme
%       idx:    index set corresponding to a dense cluster of G
%       omega:  cluster size estimate used to select vertices in idx
%
% For more details, please see the article
%   P.C. Lusk, K. Fathian, J.P. How, "CLIPPER: A Graph-Theoretic Framework
%       "for Robust Data Association," ICRA 2021
%
function [u, idx, omega] = clipper(M, C, u0inp)
%% Check input format

n = length(M); % input size

if norm2(M-M') > 1e-6 
    error('The affinity matrix M is not symmetric.');
end
if norm2(C-C') > 1e-6 
    error('The constraint matrix C is not symmetric.');
end
if n ~= length(C)
    error('Matrices M and C are not of the same size.');
end
if (max(C(:)) > 1) || (min(C(:)) < 0)
    error('The constraint matrix C is not binary');
end
if (max(M(:)) > 1) || (min(M(:)) < 0)
    error('Entries of affinity matrix M are not between 0-1');
end

%% Set parameters

% Stopping criteria
tolu = 1e-8; % stop when changes of gradient vector or optimization variable between steps
tolF = 1e-9; % 
% tolFop = 1e-10; % stop grad ascent loop if orthog proj of gradient is nearly zero
beta = 0.25;  % for reduction of alpha based on backtracking line search
maxlsiters = 99; % maximum number of line search iterations
maxiniters = 200; % maximum number of gradient ascent steps for each d
maxoliters = 1000; % maximum number of outer loop iterations
epsilon = 1e-9; % numerical threshold to replace 0 with 100x machine epsilon

%% Initial steps

% if any node should not be selected, set its corresponding affinity scores to zero
idxC = (diag(C)==0);
if any(idxC) 
    C(idxC,:) = 0;
    C(:,idxC) = 0;
end

M = M .* C; % Hadamard product of M and C to zero out any entry of M with active constraint
Cb = ones(n) - C;  % binay complement of constraint matrix C

% Initial guess (default: rand(n,1))
if nargin == 3
    u0 = u0inp;
else
    u0 = rand(n,1); % random non-negative vector (randomness breaks saddle points)
end

% One step of power method to have a good scaling of u
u = M * u0;
u = u ./ norm2(u);

% Initial d 
Cbu = Cb * u;
idxD = (Cbu > epsilon) & (u > epsilon); % indices where Cbu>0 and u>0
if any(idxD)
    Mu = M * u;
    d = mean( Mu(idxD)./Cbu(idxD) ); % smallest d that makes ?u= Mu - d Cbu  hit the positive orthant boundaries (for indices where Cbu>0 and u>0)
else
    d = 0; % in this case matrix Cb is all zeros (or there’s no active constraint)
end

Md = M - d * Cb;

%% Projected gradient ascent loop

i = 1;
outerloop = true;
while outerloop && i<maxoliters
    F = u.' * Md * u; % initial objective value
    
    j = 1;
    innerloop = true;
    while innerloop && j<maxiniters
        gradF = Md * u; % gradient
        gradFop = gradF;
%         gradFop = gradF - gradF.'*u * u;
%         if norm2(gradFop) < tolFop, break; end
%
%         % Identify an aggressive initial step size based on which elements
%         % the gradient indicates can be penalized (unless already negative)
%         idxA = (gradFop < -epsilon) & (u > epsilon);
%         if any(idxA) % if set is not empty
%             alpha = min(abs( u(idxA)./gradFop(idxA) )); % Find smallest alpha that leads to a step that hits the positive orthant boundaries (for indices where gradF<0 and u>0)
%         else % choose a large alpha (in the limit alpha -> +\infty we get u -> gradF / ||gradF|| in the following loop)
%             alpha = (1/beta)^3 / norm2(gradFop); % (1/beta)^3 = 64, which gives 3 steps of backtracking line search if step is too large before gradu becomes norm 1
%         end

        alpha = 1;
        
        k = 1;
        while true && k<maxlsiters
            unew = u + alpha * gradFop;    % gradient step
            unew = max(unew, 0);           % Projection onto positive orthant
            unew = unew ./ norm2(unew);    % normalize
            Fnew = unew.' * Md * unew;     % new objective value
            deltaF = Fnew - F;             % change in objective value
            
            if deltaF < -epsilon % if objective value has decreased (the goal is to increase)
                alpha = alpha * beta; % reduce the step size by beta (backtracking line search)
            else
                break % break from line search
            end
            
            k = k + 1;
        end

        deltau = norm2(unew - u); % change in vector u
        if ( (deltau < tolu) || (abs(deltaF) < tolF) ) %&& (~any(idxA)) % last condition is to ensure we don't break loop on an iteration that 'u' hits the positive orthant boundary (due to numerical reasons)
            innerloop = false; % if desired accuracy reached break from inner loop
        end
        
        % update values
        F = Fnew;
        u = unew;
        
        j = j + 1;
    end
    
    % Increase d
    Cbu = Cb * u;
    idxD = (Cbu > epsilon) & (u > epsilon); % indices where Cbu>0 and u>0
    if any(idxD)
        Mu = M * u;
        deltad = mean(abs( Mu(idxD)./Cbu(idxD) )); % Find smallest deltad that makes gradF=Mu-dCbu  hit the positive orthant boundaries (for indices where Cbu>0 and u>0)
        
        d = d + deltad; % increase d
        Md = M - d * Cb; % update matrix Md
    else
        outerloop = false; % break from the outer loop
    end

    i = i + 1;
end
        

%% Generate output

omega = round(u.' * Md * u); % estimate for cluster size based on the largest eigenvalue

[~, srtIdx] = sort(u, 'descend'); % sort elements of u
idx = srtIdx(1:omega); % index of 'omega' largest elements of 'u'

%% l2 norm of vector (or vectorized matrix) 
function nrm = norm2(V)
nrm = sqrt( (V(:)).' * V(:) );
