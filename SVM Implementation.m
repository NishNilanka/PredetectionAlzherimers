clc
clear all

% Load Datasets
Dataset = 'C:\Users\Nishan Gunawardena\Desktop\Abstract\PNG\Dataset';   
Testset  = 'C:\Users\Nishan Gunawardena\Desktop\Abstract\PNG\TestSet';

% we need to process the images first.
% Convert images into grayscale
% Resize the images
width=160; height=160;
DataSet = cell([], 1);

 for i=1:length(dir(fullfile(Dataset,'*.png')))

     % Training set process
     k = dir(fullfile(Dataset,'*.png'));
     k = {k(~[k.isdir]).name};
     for j=1:length(k)
        tempImage       = imread(horzcat(Dataset,filesep,k{j}));
        imgInfo         = imfinfo(horzcat(Dataset,filesep,k{j}));

         % Image transformation
         if strcmp(imgInfo.ColorType,'grayscale')
             % array of images
            DataSet{j}   = double(imresize(tempImage,[width height]));
         else
             % array of images
            DataSet{j}   = double(imresize(rgb2gray(tempImage),[width height]));
         end
     end
 end
TestSet =  cell([], 1);
  for i=1:length(dir(fullfile(Testset,'*.png')))

     % Training set process
     k = dir(fullfile(Testset,'*.png'));
     k = {k(~[k.isdir]).name};
     for j=1:length(k)
        tempImage       = imread(horzcat(Testset,filesep,k{j}));
        imgInfo         = imfinfo(horzcat(Testset,filesep,k{j}));

         % Image transformation
         if strcmp(imgInfo.ColorType,'grayscale')
             % array of images
            TestSet{j}   = double(imresize(tempImage,[width height])); 
         else
             % array of images
            TestSet{j}   = double(imresize(rgb2gray(tempImage),[width height])); 
         end
     end
  end

% Prepare class label for first run of svm
train_label               = zeros(size(754,1),1);
train_label(1:410,1)   = 1;         % 1 = AD 
train_label(410:754,1)  = 2;         % 2 = Healthy

% Prepare numeric matrix for svmtrain
Training_Set=[];
for i=1:length(DataSet)
    Training_Set_tmp   = reshape(DataSet{i},1, 100*100);
    Training_Set=[Training_Set;Training_Set_tmp];
end

Test_Set=[];
for j=1:length(TestSet)
    Test_set_tmp   = reshape(TestSet{j},1, 100*100);
    Test_Set=[Test_Set;Test_set_tmp];
end

SVMStruct = svmtrain(Training_Set ,  train_label, 'kernel_function', 'rbf');
Group = svmclassify(SVMStruct, Test_Set,'ShowPlot',true)
hold on;
plot(Test_Set(:,1),Test_Set(:,2),'ro','MarkerSize',12);
hold off