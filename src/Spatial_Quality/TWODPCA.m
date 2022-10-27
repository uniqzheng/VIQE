% This code implments classical 2DPCA for face recognition.  
% You can use any dataset.

clc
clear all;
load ORL_FaceDataSet;  % Loading face dataset. ORL consists of 40 classes, each comprising 10 samples
A=double(ORL_FaceDataSet);

%  Specifying the numbers of training and testing samples, and also the
%  number of eigenvectors (DIM) 
%-----------------------------------------------------------------------
Num_Class=40;
No_SampleClass=10;
No_TrainSamples=5;
No_TestSamples=5;
DIM=6;    % DIM can be changed form 1 to n

% Separating the dataset into training and testing sets, and then labeling.    
%-------------------------------------------------------------------------------------------
[TrainData, TestData]=Train_Test(A,No_SampleClass,No_TrainSamples,No_TestSamples);
[m,n,TotalTrainSamples] = size(TrainData);
[m1,n1,TotalTestSamples] = size(TestData);
[TrainLabel,TestLabel]=LebelSamples(Num_Class, No_TrainSamples, No_TestSamples);

% Computing image covariance (scatter) matrix
%-----------------------------------------------------------------------------
TrainMean = mean(TrainData,3); % Total mean of the training set
Gt=zeros([ n n]);
for i=1:TotalTrainSamples
    Temp = TrainData(:,:,i)- TrainMean;
    Gt = Gt + Temp'*Temp;
end
Gt=Gt/TotalTrainSamples; 

% Applying eigen-decompostion to Gt and returning transformation matrix
% 
%---------------------------------------------------------------------------------
[EigVect1,EigVal1]=eig_decomp(Gt);
EigVect=EigVect1(:,1:DIM); 

% Deriving training feature matrices
%----------------------------------------------------------------------------------

for i=1:TotalTrainSamples
    Ytrain(:,:,i)=TrainData(:,:,i)*EigVect;
end

% Testing and Classification
%----------------------------------------------------
TestResult = zeros(TotalTestSamples,1);


for i=1:TotalTestSamples
    
    Distance = zeros(TotalTrainSamples,1);
    
    Ytest = TestData(:,:,i)* EigVect; % Deriving test feature matrix
   
    for j=1:TotalTrainSamples
        for k=1:DIM
            Distance(j) = Distance(j) + norm(Ytest(:,k)-Ytrain(:,k,j)); % Measuring the distances between test and training feature matrices 
        end
    end
    
    [MINDIST ID] = min(Distance); % Returning Min distance
    TestResult(i) = TrainLabel(ID);
    % Lines 68 to 75 can be uncommented to speed the process   
    subplot 221; imshow(TestData(:,:,i),[]);title(['Tested Face = ' num2str(i)]);
    subplot 222; imshow(TrainData(:,:,ID),[]);title(['Recognized Face = ' num2str(ID)]);
    subplot(2,2,[3 4]); plot(Distance,'-o','MarkerIndices',[ID ID],'MarkerFaceColor','blue','MarkerSize',5);title(['Min Distance = ' num2str(MINDIST),' ID = ' num2str(ID)]);
    xlabel('Training Samples') 
    ylabel('Distance') 
    grid on
    grid minor
    pause (0.01) %  You can delay the process as you wish;
end
Result = (TestResult == TestLabel);

CorrectRate = 100*(sum(Result/TotalTestSamples))

