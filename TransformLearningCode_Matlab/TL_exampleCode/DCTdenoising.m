function [IMU, paramsout]= DCTdenoising(I1,I7,paramsin)

%This is an implementation of the transform learning based denoising method used in the simulations in the following paper:
%1) S. Ravishankar and Y. Bresler, “\ell_0 Sparsifying transform learning with efficient optimal updates and convergence guarantees,” IEEE Transactions on Signal Processing, vol. 63, no. 9, pp. 2389-2404, May 2015.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Inputs: 1) I1 : Noisy image
%        2) I7 : Noiseless reference
%        3) paramsin: Structure that contains the parameters of the denoising algorithm. The various fields are as follows -
%                   - sig: Standard deviation of the i.i.d. Gaussian noise
%                   - iterx: Number of iterations of the two-step denoising algorithm (e.g., iterx= 11)
%                   - n: Patch size, i.e., Total number of pixels in a square patch (e.g., n= 121, 64)
%                   - N: Number of training signals used in the transform learning step of the algorithm (e.g., N=500*64)
%                   - C: Parameter that sets the threshold that determines sparsity levels in the variable sparsity update step (e.g., C=1.04 when n=121; C = 1.08 when n=64)
%                   - s: Initial sparsity level for patches (e.g., s=round((0.1)*n))
%                   - tau: Sets the weight \tau in the algorithm (e.g., tau=0.01/sig)
%                   - maxsparsity: Maximum sparsity level allowed in the variable sparsity update step of the algorithm (e.g., maxsparsity = round(6*s))
%                   - M: Number of iterations within transform learning step (e.g., M = 12)
%                   - method: If set to 0, transform learning is done employing a log-determinant+Frobenius norm regularizer.
%                             For any other setting, an orthonormal transform learning procedure is adopted. (e.g., method=0)
%                   - lambda0: Determines the weight on the log-determinant+Frobenius norm regularizer. To be used in the case when method = 0. (e.g., lambda0=0.031)
%                   - W: Initial transform in the algorithm  (e.g., W=kron(dctmtx(sqrt(n)),dctmtx(sqrt(n))))
%                   - r: Patch Overlap Stride (e.g., r=1)
%

%Outputs:  1) IMU: Denoised image
%          2) paramsout - Structure containing outputs other than the denoised image.
%                 - PSNR : PSNR of denoised image.
%                 - transform : learnt transform

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initializing algorithm parameters
[aa,bb]=size(I7);
sig = paramsin.sig;
n = paramsin.n;
C1 = paramsin.C;
la = paramsin.tau;
maxsp = paramsin.maxsparsity;
W =  paramsin.W;
r = paramsin.r;

threshold=C1*sig*(sqrt(n)); %\ell_2 error threshold (maximum allowed norm of the difference between a noisy patch and its denoised version) per patch

%Initial steps

%Extract image patches
[TE,idx] = my_im2col(I1,[sqrt(n),sqrt(n)],r); br=mean(TE);
TE=TE - (ones(n,1)*br); %subtract means of patches
[rows,cols] = ind2sub(size(I1)-sqrt(n)+1,idx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        X1=W*TE;   %In the last iteration of the two-step denoising algorithm, apply the transform to all patches
        kT=zeros(1,size(TE,2));
    
    [~,ind]=sort(abs(X1),'descend');
    er=n*(0:(size(X1,2)-1));ind=ind + (er'*ones(1,n))';
    G=(pinv([(sqrt(la)*eye(n));W])); Ga=G(:,1:n);Gb=G(:,n+1:2*n);
    
    %Variable Sparsity Update Step
        Gz=Ga*((sqrt(la))*TE);
        q=Gz;   %In the last iteration of the two-step denoising algorithm, q stores the denoised patches.
        ZS2=sqrt(sum((Gz-TE).^2));
        kT=kT+(ZS2<=threshold);  %Checks if error threshold is satisfied at zero sparsity for any of the patches
        STY=zeros(1,size(TE,2));X=zeros(n,size(TE,2)); %STY is a vector of sparsity levels and X is the corresponding sparse code matrix
    
    %Incrementing sparsity by 1 at a time, until error threshold is satisfied for all patches.
    for k=1:maxsp
        indi=find(kT==0); %Find indices of patches for which the error threshold has not yet been satisfied
        if(isempty(indi))
            break;
        end
        
        X(ind(k,indi))=X1(ind(k,indi));  %Update sparse codes to the current sparsity level in the loop.
            q(:,indi)= Gz(:,indi) + Gb*(X(:,indi));  %Update denoised patches in the last iteration of the two-step denoising algorithm
            ZS2=sqrt(sum((q(:,indi) - TE(:,indi)).^2)); kT(indi)=kT(indi)+(ZS2<=threshold);  %Check if error threshold is satisfied at sparsity k for any patches
        STY(indi)=k;  %Update the sparsity levels of patches
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Averaging together the denoised patches at their respective locations in the 2D image
IMout = zeros(aa,bb);
Weight=zeros(aa,bb);
bbb=sqrt(n);
for jj = 1:10000:size(TE,2)
    jumpSize = min(jj+10000-1,size(TE,2));
    ZZ= q(:,jj:jumpSize) + (ones(size(TE,1),1) * br(jj:jumpSize));
    inx=(ZZ<0);ing= ZZ>255; ZZ(inx)=0;ZZ(ing)=255;
    for ii  = jj:jumpSize
        col = cols(ii); row = rows(ii);
        block =reshape(ZZ(:,ii-jj+1),[bbb,bbb]);
        IMout(row:row+bbb-1,col:col+bbb-1)=IMout(row:row+bbb-1,col:col+bbb-1)+block;
        Weight(row:row+bbb-1,col:col+bbb-1)=Weight(row:row+bbb-1,col:col+bbb-1)+ones(bbb);
    end;
end
IMU = (IMout)./(Weight);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ing= IMU<0; ing2= IMU>255;
IMU(ing)=0;IMU(ing2)=255;  %Limit denoised image pixel intensities to the range [0, 255].

paramsout.PSNR=20*log10((sqrt(aa*bb))*255/(norm(double(IMU)-double(I7),'fro')));
end
