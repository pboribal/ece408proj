%This demo code simulates a transform learning based image denoising example presented in the following paper:
%1) S. Ravishankar and Y. Bresler, “\ell_0 Sparsifying transform learning with efficient optimal updates and convergence guarantees,” 
%   IEEE Transactions on Signal Processing, vol. 63, no. 9, pp. 2389-2404, May 2015.

addpath('./DatausedforDenoisingExperiment/Barbara/sigma5')

load I1  %noisy image
load I7  %noiseless reference

sig=5;  %standard deviation of Gaussian noise
n=121;  %patch size (number of pixels in patch) to use (e.g., n = 121, 64).

s=round((0.1)*n);
if(sig>=100)
    iter=5;
else
    iter=11;
end

%initialize parameters
paramsin.sig = sig;
paramsin.iterx = iter;
paramsin.n = n;
paramsin.N = 32000;
if(n==64)
    C=1.08;
end
if(n==121)
    C=1.04;
end
paramsin.C = C;
paramsin.tau = 0.01/sig;
paramsin.s = s;
paramsin.M = 12;
paramsin.maxsparsity = round(6*s);
paramsin.method = 0;
paramsin.lambda0 = 0.031;
paramsin.W = kron(dctmtx(sqrt(n)),dctmtx(sqrt(n)));
paramsin.r = 1;

%Denoising algorithm
[IMU,paramsout]= TSPCLOSEDFORMdenoising(I1,I7,paramsin); %IMU is the output denoised image