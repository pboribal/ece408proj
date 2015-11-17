function [W,X]= transformLearning(W, Y, numiter, sparsityList)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Inputs: 1) W : Initial Transform
%        2) Y : Training Matrix with signals as columns
%        3) numiter:  Number of iterations of alternating minimization
%        4) sparsityList: Vector containing maximum allowed sparsity levels for each training signal.

%Outputs:  1) W: Learnt Transform
%          2) X: Learnt Sparse Code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initial steps
[K, n] = size(W);                                    % measure transform size
ix = find(sparsityList>0);                          %
q = Y(:,ix);                                        %
sparsityList = sparsityList(:,ix);                  % exclude 'zero-sparsity' data
N = size(q,2);                                      % effective data size
ez = K*(0:(N-1));                                   %
sparsityList = sparsityList + ez;                   % weighted sparsity
%Algorithm iterations in a FOR Loop
for i = 1 : numiter
    %Sparse Coding Step
    X1 = W * q;                                     % matrix-vector multiply
    [s] = sort(abs(X1),'descend');                  % sorting
    X = X1.*(bsxfun(@ge,abs(X1),s(sparsityList)));  % keep s-largest non-zero  
    %Transform Update Step
    [U,~,V]=svd(q*X');                              % SVD of YX'
    W=(V(:,1:n))*U';                                % Matrix multiplication
end