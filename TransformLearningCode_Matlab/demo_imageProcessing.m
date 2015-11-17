%This demo code simulates a transform learning example presented in Figure 3 of the following paper:
%1)S. Ravishankar and Y. Bresler, “\ell_0 Sparsifying transform learning with efficient optimal updates and convergence guarantees,” 
%  IEEE Transactions on Signal Processing, vol. 63, no. 9, pp. 2389-2404, May 2015.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Inputs: 
%        1): input: aa * bb gray-scale image, to be compressed

%Outputs: 1): output: aa * bb decompressed image

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
input=imread('barbara.png');               % load image
[aa,bb]=size(input);                       % measure size
input=(double(input));                        % convert to double
    
%%%%%%%%%%% Set parameters for transform learning %%%%%%%%%%%%%%%%%%%%%%%%%
n=64;                                   % patch size (number of pixels in patch)
T0=round((11/64)*n);                    % sparsity level for each patch, K
lambda0=3.1e-3;  %parameter that sets the weights on the log-determinant and Frobenius norm penalties in problem formulation
numt=100;  %number of iterations of alternating transform learning algorithm (i.e., iterations of sparse coding and transform update)
W0=kron(dctmtx(sqrt(n)),dctmtx(sqrt(n))); % initialization for the transform: 2D DCT 

%%%%%%%%%%%% Main Program %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stride = sqrt(n);   % non-overlapping
stride = 1;        % overlapping - choose either one

%extract non-overlapping image patches and subtract the means of the patches
[data, idx] = my_im2col(input,[sqrt(n),sqrt(n)], stride);       % break the image into vectorized patches
[rows,cols] = ind2sub(size(input)-sqrt(n)+1,idx);               % indexing the patch row/column number
N = size(data, 2);                                              % total data amount

br=mean(data);                    %
data = data - (ones(n,1)*br);         % de-mean (optional)

% <<Transform learning>>
%The outputs below are - W: the learnt transform
%                        X: matrix of transform sparse representations.
STY=T0*ones(1,size(data,2));          % sparsity (K) for each patch
[W,X]= transformLearning(W0,data,numt,STY);

% <<Image Reconstruction>>
q = W\X;                            % W^{-1} * X
IMout=zeros(aa,bb);                 % initialize the image
Weight=zeros(aa,bb);                % initialize the weight (for overlapping)
bbb = sqrt(n);
%%%%% the following codes are to speed up the Matlab for-loop %%%%
% for jj = 1:10000:size(TE,2)
%     jumpSize = min(jj+10000-1,size(TE,2));
%     ZZ= q(:,jj:jumpSize) + (ones(size(TE,1),1) * br(jj:jumpSize));
%     inx=(ZZ<0);ing= ZZ>255; ZZ(inx)=0;ZZ(ing)=255;
%     for ii  = jj:jumpSize
%         col = cols(ii); row = rows(ii);
%         block =reshape(ZZ(:,ii-jj+1),[bbb,bbb]);
%         IMout(row:row+bbb-1,col:col+bbb-1)=IMout(row:row+bbb-1,col:col+bbb-1)+block;
%         Weight(row:row+bbb-1,col:col+bbb-1)=Weight(row:row+bbb-1,col:col+bbb-1)+ones(bbb);
%     end;
% end
q = q + ones(n,1) * br;             % bring the mean back to vectorized patches
% loop for each patch
for patchNum = 1 : N
    col = cols(patchNum);           % current patch column number
    row = rows(patchNum);           % current patch row number
    block = reshape(q(:, patchNum) ,[bbb,bbb]);     % bring the vector into bbb * bbb patch
    IMout(row:row+bbb-1,col:col+bbb-1) = IMout(row:row+bbb-1,col:col+bbb-1) + block;        % add the patch to the perspetive location
    Weight(row:row+bbb-1,col:col+bbb-1) = Weight(row:row+bbb-1,col:col+bbb-1) + ones(bbb);  % add the weight to the perspetive location
end
output=(IMout)./(Weight);           % weighted for the overlappings

%%%%%%%%%%%%%% display the image %%%%%%%%%%%%%%
imshow(input, [0 255]);
figure; imshow(output, [0 255]);
figure; imshow(abs(input - output),[]);
colormap('jet');