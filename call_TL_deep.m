close all
clear all
clc

%%
%data

load mnist_basic.mat;

%%
% X_train= (Train.X)';
% label_train= Train.y;
% X_test= (Test.X)';
% label_test= Test.y;
%%
%MNIST_variations
% load mnistbackground_rand.mat;
% load mnist_background_rotated.mat;
% load mnist_backimg_thr_new.mat;
% load rot_backimage_threshold.mat;
% load rot_mnistdata.mat;
%%
%USPS
% load usps.mat
%%
%Bangla
% load banglatraindata.mat;
% load bangtestdata.mat;
%%
%Devnagri
% load Devnagritraindata.mat;
% load Devnagritestdata.mat;
% X_train= trn.X;
% label_train= trn.y;
% X_test= tst.X;
% label_test= tst.y;

%%
% 
% load YaleB.mat;
% % 
% A= train_lbl;
% B= repmat(1:38,1216,1);
% label_train= sum(A.*B');
% label_train= label_train';
% 
% A= test_lbl;
% B= repmat(1:38,1198,1);
% label_test= sum(A.*B');
% label_test= label_test';
% 
% X_train= train_img;
% X_test= test_img;

%%
% load AR.mat;
% % 
% A= train_lbl;
% B= repmat(1:100,2000,1);
% label_train= sum(A.*B');
% label_train= label_train';
% 
% A= test_lbl;
% B= repmat(1:100,600,1);
% label_test= sum(A.*B');
% label_test= label_test';
% 
% X_train= train_img;
% X_test= test_img;


%%
% %load mnist_full
% X_train = loadMNISTImages('train-images.idx3-ubyte'); % train_image_matrix
% label_train = loadMNISTLabels('train-labels.idx1-ubyte');    % test_image_matrix
% X_test = loadMNISTImages('t10k-images.idx3-ubyte'); % test_image_matrix
% label_test = loadMNISTLabels('t10k-labels.idx1-ubyte'); %test_image_labels


%%

%load cifar-10
% load inputimage.mat;
% load labelout.mat;
% load test_batch.mat;

% X_train= double(inputimage');
% label_train= double(labelout);
% X_test= double(data');
% label_test= double(labels);


%%

numOfAtoms1= 392;
numOfAtoms2= 196;
numOfAtoms3= 98;

%%
%one-layer
% [T1,  Z1, lambda] = TransformLearning_deep (X_train, numOfAtoms1);
% T= T1;
% Z_train= Z1;

%two-layers
% [T1, T2,  Z1, Z2, lambda] = TransformLearning_deep (X_train, numOfAtoms1,numOfAtoms2);
% T= T2*T1;
% Z_train= Z2;
% % 

%three-layers
[T1, T2, T3, Z1, Z2, Z3, lambda] = TransformLearning_deep (X_train, numOfAtoms1,numOfAtoms2, numOfAtoms3);
T= T3*T2*T1;
Z_train= Z3;


%%
% % Testing
Z_test = sign(T*X_test).*max(0,abs(T*X_test)-lambda);   


%%
% % Classification (KNN)
label = label_train+1;
testlabels = label_test+1;
%
knnmodel = fitcknn (Z_train', label);

PredLabels = predict(knnmodel, Z_test');
% 
disp('Out of Classifier!');
correct = length(find(PredLabels==testlabels));
percent=correct/length(testlabels) * 100

%%

% %construct a binary SVM classifier
opts = optimset('MaxIter',2000,'TolX',5e-4,'TolFun',5e-4);

svmModel = cell(1,max(label));
for ii = min(label):max(label)
%     fprintf('Learning SVM classifier for Class %d ... ',ii)
    currentClass = Z_train(:,label==ii);
    negClass = Z_train(:,label~=ii);
    posLabel = 1*ones(1,size(currentClass,2));
    negLabel = -1*ones(1,size(negClass,2));
    Xtr = [currentClass,negClass];
    Ltr = [posLabel,negLabel];
    t = randperm(size(Xtr,2));
    Xtr = Xtr(:,t);
    Ltr = Ltr(1,t);
    svmModel{1,ii} = fitcsvm(Xtr',Ltr,'Standardize',true,'KernelFunction','rbf','KernelScale','auto');
%     fprintf('SVM Learnt \n');
end
Scores = zeros(size(Z_test,2),numel(unique(label_test)));
for class = min(testlabels):max(testlabels)
    [~,score] = predict(svmModel{1,class},Z_test');
    Scores(:,class) = score(:,2);
end
[~,maxScore] = max(Scores,[],2); %maxScore is px1
if size(label_test,2) >1
    testlabels = testlabels';
end
accSVM = length(find((maxScore) == testlabels))/length(label_test);
fprintf('Accuracy = %d',accSVM*100)

%%
% display transform at first layer
rx= randperm(200);
Tx= T1(rx(1:36), :) ;
% Tx= T1(1:36, :);
A= ones(180,180);
k=1;
for i= 1: 6
    for j=1:6
       A(30*(i-1)+1: 30*(i-1)+28,30*(j-1)+1: 30*(j-1)+28 )=  reshape(Tx(k,:),28,28);
       k=k+1;
    end
end
imshow (A)
%%