function [T1,  T2, T3, Z1, Z2, Z3, lambda] = TransformLearning_deep (X, numOfAtoms1,numOfAtoms2, numOfAtoms3)
% function [T1, T2,  Z1, Z2, lambda] = TransformLearning_deep (X, numOfAtoms1,numOfAtoms2)
% function [T1,  Z1, lambda] = TransformLearning_deep (X, numOfAtoms1)

% 1-layer solves ||TX - Z||_Fro - mu*logdet(T) + eps*mu||T||_Fro + lambda||Z||_1

% Inputs
% X          - Training Data
% numOfAtoms - dimensionaity after Transform
% mu         - regularizer for Tranform
% lambda     - regularizer for coefficient
% eps        - regularizer for Transform
% type       - 'soft' or 'hard' update: default is 'soft'
% Output
% T          - learnt Transform
% Z          - learnt sparse coefficients


%%
%set params
% lambda= 0.01;
% epsilon= 0.1;
% mu = 100;
mu1=100;
mu2=100;
mu3=100;
eps1=.1;
eps2=.1;
eps3=.1;
lambda=0.01;

maxIter = 5;
type = 'soft'; % default 'soft'

%%
%initializations
rng(1); % repeatable
T1 = randn(numOfAtoms1, size(X,1));

invL1 = (X*X' + mu1*eps1*eye(size(X,1)))^(-0.5);

    switch type
        case 'soft'
            Z1 = sign(T1*X).*max(0,abs(T1*X)-lambda); % soft thresholding
        case 'hard'
            Z1 = (abs(T1*X) >= lambda) .* (T1*X); % hard thresholding
    end


T2 = randn(numOfAtoms2, size(Z1,1));

invL2 = (Z1*Z1' + mu2*eps2*eye(size(Z1,1)))^(-0.5);

    switch type
        case 'soft'
            Z2 = sign(T2*Z1).*max(0,abs(T2*Z1)-lambda); % soft thresholding
        case 'hard'
            Z2 = (abs(T2*Z1) >= lambda) .* (T2*Z1); % hard thresholding
    end

T3 = randn(numOfAtoms3, size(Z2,1));

invL3 = (Z2*Z2' + mu3*eps3*eye(size(Z2,1)))^(-0.5);

    switch type
        case 'soft'
            Z3 = sign(T3*Z2).*max(0,abs(T3*Z2)-lambda); % soft thresholding
        case 'hard'
            Z3 = (abs(T3*Z2) >= lambda) .* (T3*Z2); % hard thresholding
    end

%%
% update steps

for i = 1:maxIter
    
%     update Coefficient Z sparse 
    switch type
        case 'soft'
            Z1 = sign(T1*X).*max(0,abs(T1*X)-lambda); % soft thresholding
        case 'hard'
            Z1 = (abs(T1*X) >= lambda) .* (T1*X); % hard thresholding
    end

    [U,S,V] = svd(invL1*X*Z1');
    D1 = [diag(diag(S) + (diag(S).^2 + 2*mu1).^0.5) zeros(numOfAtoms1, size(X,1)-numOfAtoms1)];
    T1 = 0.5*V*D1*U'*invL1;
    
    switch type
        case 'soft'
            Z2 = sign(T2*Z1).*max(0,abs(T2*Z1)-lambda); % soft thresholding
        case 'hard'
            Z2 = (abs(T2*Z1) >= lambda) .* (T2*Z1); % hard thresholding
    end


    [U,S,V] = svd(invL2*Z1*Z2');

    D2 = [diag(diag(S) + (diag(S).^2 + 2*mu2).^0.5) zeros(numOfAtoms2, size(Z1,1)-numOfAtoms2)];
% 
T2 = 0.5*V*D2*U'*invL2;
    
    switch type
        case 'soft'
            Z3 = sign(T3*Z2).*max(0,abs(T3*Z2)-lambda); % soft thresholding
        case 'hard'
            Z3 = (abs(T3*Z2) >= lambda) .* (T3*Z2); % hard thresholding
    end

    [U,S,V] = svd(invL3*Z2*Z3');
    D3 = [diag(diag(S) + (diag(S).^2 + 2*mu3).^0.5) zeros(numOfAtoms3, size(Z2,1)-numOfAtoms3)];
    T3 = 0.5*V*D3*U'*invL3;
% % % %  

lambda= lambda*0.1;
% mu1 = mu1*0.5;
% mu2 = mu2*0.5;
% mu3 = mu3*0.1;
% eps1 = eps1*0.1;
% eps2 = eps2*0.5;
% eps3 = eps3*0.1;
  
end
