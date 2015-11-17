function [x] = sparseCode_Klargest(y, K)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs: 
%        2) y : non-sparse vector
%        3) K : number of elemnets kept

%Outputs: 2) x: sparse Code

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[s]=sort(abs(y),'descend');                % sorting
x = y.*(bsxfun(@ge,abs(y), s(K)));     % keep s-largest non-zero  
end
