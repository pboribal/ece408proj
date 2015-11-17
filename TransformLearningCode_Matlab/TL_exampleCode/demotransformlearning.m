%This demo code simulates a transform learning example presented in Figure 3 of the following paper:
%1)S. Ravishankar and Y. Bresler, “\ell_0 Sparsifying transform learning with efficient optimal updates and convergence guarantees,” 
%  IEEE Transactions on Signal Processing, vol. 63, no. 9, pp. 2389-2404, May 2015.

clear all;
I1=imread('cameraman.png'); %read image
[aa,bb]=size(I1);
I1=(double(I1));

%%%%%%%%%%%Set parameters for transform learning%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=64; %patch size (number of pixels in patch)
T0=round((11/64)*n);  %sparsity level for each patch
lambda0=3.1e-3;  %parameter that sets the weights on the log-determinant and Frobenius norm penalties in problem formulation
numt=600;  %number of iterations of alternating transform learning algorithm (i.e., iterations of sparse coding and transform update)
W0=kron(dctmtx(sqrt(n)),dctmtx(sqrt(n))); %2D DCT initialization for the transform

%Note that in general, the parameters above such as lambda0 or the transform initialization, etc., may need to be carefully chosen so as to achieve a certain condition number 
%for the learnt transform, or to ensure quick algorithm convergence, etc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%extract non-overlapping image patches and subtract the means of the patches
[blocks] = my_im2col(I1,[sqrt(n),sqrt(n)],sqrt(n)); br=mean(blocks); 
TE=blocks - (ones(n,1)*br);

YH=TE; %Training set

%Transform learning. The outputs below are - W: the learnt transform, and X: matrix of transform sparse representations.
STY=T0*ones(1,size(YH,2)); l2=lambda0*((norm(YH,'fro'))^2); l3=l2;
[W,X]= TLclosedformmethod(W0,YH,numt,l2,l3,STY);