function [X] = sparseCode_Proj(W, Y, sparsityList )
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs: 1) W : Initial Transform
%        2) Y : Training Matrix with signals as columns
%        3) sparsityList: Vector containing maximum allowed sparsity levels for each training signal.

%Outputs: 2) X: Learnt Sparse Code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1 = W * Y;                                 % Matrix-Vector Multiplication
[K, ~] = size(X1);                          % data dimensionality
ix = find(sparsityList>0);
a0 = X1(:,ix); 
sparsityList=sparsityList(:,ix);            % (optional) exclude 'zero-sparsity' data    
N=size(a0,2);
ez=K*(0:(N-1));
sparsityList = sparsityList + ez;
[s]=sort(abs(a0),'descend');                % sorting
a1 = a0.*(bsxfun(@ge,abs(a0),s(sparsityList)));     % keep s-largest non-zero  
X = zeros(size(X1));
X(:,ix) = a1;                               % rescale back to original size
end

