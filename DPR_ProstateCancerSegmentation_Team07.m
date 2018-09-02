%Detection and Pattern Recognition Lab Task 1
%Prostate Cancer Segmentation
%---------------------------------------------------------------------
%The following code contains the entire pipeline of the task, Set the
%conditions before running the program.

%% Parameters and pipeline initial conditions
% Set true or false to run the following sections:

%1. Extract Prostate region
UseExtractProstateCode = false; 

%2. Feature Normalisation
% Three Methods : 1.Variance 2.MinMax 3.UnitVector
UseNormalisationCode = true;
NormalisationMethod = 'Variance' ; 

%3. Outlier detection and removal
UseOutlierCode = true;

%4. Dataset partitioning
UseDataSetPartCode = true;

%5. Trainingset selection
UseTrainingSelectionCode = true;
%1.frequency 2.Random 3.RandomSelction
TypeOfSelection = 'RandomSelction';

%6. Classifiers : 1.nearest-mean 2.kNN 3.SVM
UseClassifierCode = true;

%a. nearest-mean parameters
%b. kNN parameters
Neighbours = 15;
ExpertCol = 3; %ExpertA
%c. SVM parameters
classifer = 'SVM';

%7. Visualisation
UseVisualisationCode = true;

fprintf('Pipeline steps :\n');


%% Obtain ProstateArray from dataset
if UseExtractProstateCode == true
if exist('dataset','var') == 0
    error('dataset is missing from the workspace. Please load it!');
end
L=dataset;
S=size(L);
count=1;
for m=0:S(1)-1
   %inside a patient
   P=L{m+1,1};
   M=size(P.LabelsA);
   %filter labels that is non zero
   
   
   for k=0:M(3)-1
       for j=0:M(2)-1
           for i=0:M(1)-1
                 if ((P.LabelsA(i+1,j+1,k+1)~=0)||(P.LabelsB(i+1,j+1,k+1)~=0))
                     
                     ProstateArray(count,1)=P.Image(i+1,j+1,k+1,1);
                     ProstateArray(count,2)=P.Image(i+1,j+1,k+1,2);
                     ProstateArray(count,3)=P.Image(i+1,j+1,k+1,3);
                     ProstateArray(count,4)=P.Image(i+1,j+1,k+1,4);
                     ProstateArray(count,5)=P.Image(i+1,j+1,k+1,5);
                     ProstateArray(count,6)=P.patientIdx;
                     ProstateArray(count,7)=P.LabelsA(i+1,j+1,k+1);
                     ProstateArray(count,8)=P.LabelsB(i+1,j+1,k+1);
                     ProstateArray(count,9)= i+1;
                     ProstateArray(count,10)=j+1;
                     ProstateArray(count,11)=k+1;
                     count=count+1;
                 end
                 
           end
       end
   end
   
end
fprintf('Extracted prostate region > ');
end

%% Check whether dataset and ProstateArray are loaded into thw workspace
% Load the following data files into the workspace or make sure that the
% link to the file is correct

%dataset (given)
%ProstateArray = Raw Matrix data i.e.,452805x11 from dataset.mat

if exist('dataset','var') == 0
    error('dataset is missing from the workspace. Please load it!');
end
if exist('ProstateArray','var') == 0
    error('ProstateArray is missing from the workspace. Please load it!');
end

%% Feature Normalisation
if UseNormalisationCode == true
switch NormalisationMethod
case 'MinMax'
        s=size(ProstateArray);
        for i =1:5
            min_value=min(ProstateArray(:,i));
            max_value=max(ProstateArray(:,i));
            for c =1:s(1)
                Norm(c,i)=(ProstateArray(c,i)-min_value)/(max_value-min_value);
            end
        end
         NormalisedArray = ProstateArray;

        for i=1 : 452805
            for j = 1 : 5
            NormalisedArray(i,j) = Norm (i,j);
            end    
        end
case 'Variance'
         c=1;
         for i =1:5
             mu(i)=mean(ProstateArray(:,i));
             var1(i)=var(ProstateArray(:,i));
             for j=1:452805
             Norm1(j,i)=(ProstateArray(j,i)-mu(i))/var1(i);
             end
         end
         NormalisedArray = ProstateArray;

        for i=1 : 452805
            for j = 1 : 5
            NormalisedArray(i,j) = Norm1(i,j); %Overwrite Normalised values in the feature columns
            end    
        end
case 'UnitVector'
    NormalisedArray = ProstateArray ; 
end
fprintf('Feature Normalisation with %s > ',NormalisationMethod);
end

%% Outlier detection and removal
   Puredata = NormalisedArray;
% Check = mahal(NormalisedArray,NormalisedArray);
% Puredata = NormalisedArray;
% 
% indices = find(Check(1 : 382778,1)>72);
% Puredata(indices,:) = [];
% 
% fprintf("Outlier detection and removal > \n");

%% Dataset Partitioning with frequency feature
% Adding a new feature called frequency to the training data(1 to 11
% patients)

if UseDataSetPartCode == true    
FreqDataset = dataset; % DOnt manipulate the given data
for ID = 1:11
Patient = FreqDataset{ID,1}.Image(:,:,:,1); %Select label A
dim = size(Patient);
for k=1:dim(3)
    for j = 1:dim(2)
        for i = 1:dim(1)
            Frequency =0;
            % Find the count value of each pixel with label > 0 ... It
            % varies from 0 to 54
            if i > 1 && i < dim(1) && j > 1 && j < dim(2) && k > 1 && k < dim(3)
                Frequency = Patient(i-1,j-1,k-1)+ Patient(i-1,j-1,k) + Patient(i-1,j-1,k+1)...
                    +Patient(i,j-1,k-1)+ Patient(i,j-1,k) + Patient(i,j-1,k+1)...
                    +Patient(i+1,j-1,k-1)+ Patient(i+1,j-1,k) + Patient(i+1,j-1,k+1)...
                    +Patient(i-1,j,k-1)+ Patient(i-1,j,k) + Patient(i-1,j,k+1)...
                    +Patient(i,j,k-1)+ Patient(i,j,k) + Patient(i,j,k+1)...
                    +Patient(i+1,j,k-1)+ Patient(i+1,j,k) + Patient(i+1,j,k+1)...
                    +Patient(i-1,j+1,k-1)+ Patient(i-1,j+1,k) + Patient(i-1,j+1,k+1)...
                    +Patient(i,j+1,k-1)+ Patient(i,j+1,k) + Patient(i,j+1,k+1)...
                    +Patient(i+1,j+1,k-1)+ Patient(i+1,j+1,k) + Patient(i+1,j+1,k+1);
               
                FreqDataset{ID,1}.LabelsA(i,j,k) = Frequency / 1000;
            else
                FreqDataset{ID,1}.LabelsA(i,j,k) = Patient(i,j,k) / 1000;
            end
            
        end
    end
end
end
 % Form the TrainingData matrix with frequency
TrainingData = Puredata;
leng = size(Puredata);
for i = 1 : 382778 
    if TrainingData(i,7) > 0
        PatID = TrainingData(i,6);
        Row = TrainingData(i,9);
        Col = TrainingData(i,10);
        Slice = TrainingData(i,11);
        TrainingData(i,12) = FreqDataset{PatID,1}.LabelsA(Row,Col,Slice);
    end
end

fprintf('Dataset partioned > ');
end
% Validation Data generation
ValidationData = NormalisedArray(382779 : 452805,:);
%% Training dataset selection
ReducedTrainData = [];
if UseTrainingSelectionCode == true
    
switch TypeOfSelection
    case 'frequency'
    histogram(TrainingData(:,12)); % Plot the histogram based on frequency to determine training data selection
    % Training data reduction based on frequency
    count = 0;
    for i = 1 : 382778
        if (TrainingData(i,12) >7 && TrainingData(i,12) < 20)  || TrainingData(i,12) > 50
            count = count + 1;
            for j=1:12
            ReducedTrainData(count,j) = TrainingData(i,j) ;
            end
        end
    end
    case 'Random'
        count = 0;
        shortcount = 0;
        for i = 1 : 382778
            count = count + 1;
            if count == 10
                shortcount =shortcount + 1; 
                for j = 1: 11
                ReducedTrainData(shortcount,j) = TrainingData(i,j);
                end
                count = 0;
            end
        end
    
     case 'RandomSelction'
        ReducedTrainDataIndices = randsample(382778,38277);
        ReducedTrainData = TrainingData(ReducedTrainDataIndices,:);
end
        
fprintf('Training dataset reduced > ');
end

 
%% Classifers 
if UseClassifierCode == true
    switch classifer
        case 'nearest-mean'
            TrainCount=size(ReducedTrainData);
            ValidCount=size(ValidationData);
            %Class1
            classcount=0;
            classcount2=0;
            for i=1: TrainCount(1)
                if ReducedTrainData(i,3)==1
                    for j=1:TrainCount(2)
                        Class1Data(classcount+1,j)=ReducedTrainData(i,j);
                    end
                    classcount=classcount+1;
                end
                
            end
      
            Class1=Class1Data(:,[1 2]);
            
            %class2
            for i=1: TrainCount(1)
                if ReducedTrainData(i,3)==2
                    for j=1:TrainCount(2)
                        Class2Data(classcount2+1,j)=ReducedTrainData(i,j);
                    end
                    classcount2=classcount2+1;
                end
                
            end
            Class2=Class2Data(:,[1 2]);
            
            %Mahalonobis Distance for class1
            
            D1=mahal(ValidationData(:,[1 2]),Class1);
            
            
            %Mahalonobis Distance for class2
            
            D2=mahal(ValidationData(:,[1 2]),Class2);
            
            
            %Compare distance
            for i= 1:ValidCount(1)
                
                
                if D1(i)>= D2(i)
                    
                    labelPredicted(i,1)=2;
                else
                   
                    labelPredicted(i,1)=1;
                end
                
                
            end
            
            %calculate accuracy
            
            
            labelActual = ValidationData (:,7);
            count=1;
            for i= 1:ValidCount(1)
                if  labelPredicted(i,1)==labelActual(i,1)
                    
                    count=count+1;
                end
            end
            
            acc_nearest_mean=count/ValidCount(1)*100;
            
        case 'kNN'
            
        % No of Neighbours were selected on trail and error basis 

        % Training 1: Without frequency
        Class = fitcknn(ReducedTrainData(:,[1 2]),ReducedTrainData(:,7),'NumNeighbors',Neighbours,'Standardize',1);
        NewData = Puredata(382779:452805,:);
        labelPredicted = predict(Class,NewData(:,[1 2]));
        labelActual = ValidationData (:,7);
        Hit = 0;
        LastValue = size(ValidationData);
        NoOfCancer = size(find(ValidationData (:,7) == 2));
        for i= 1 : LastValue(1)
            if labelPredicted(i,1) == labelActual (i,1)
                if labelActual (i,1) == 2
                Hit = Hit + 1;
                end
            end
        end
        fprintf('Classifier_%s  \n',classifer);
        Accuracy_kNN = (Hit/NoOfCancer(1))*100;
      
        case 'SVM'
            %add libsvm binaries into your path
            %use >> addpath {path to libsvm binaries}
            %dividing the training and test set on the fly
            X_train = ReducedTrainData(:,[1 2]);
            Y_train = ReducedTrainData(:,7);
            NewData = Puredata(382779:452805,:);
            
            X_test = NewData(:,[1 2]);
            Y_test = NewData(:,3);
           
             model_cancer =svmtrain(Y_train, X_train, '-c 1 -g 0.1');
%             [labelPredicted, accuracy, dec_values] = svmpredict(Y_test, X_test, model_cancer);
            
            %model_cancer = fitcsvm(X_train,Y_train);
          %  svmclassify(svmStruct,Xnew,'ShowPlot',true)

           [labelPredicted, accuracy, dec_values] = svmpredict(Y_test, X_test, model_cancer);
                  %labelPredicted = predict( model_cancer , X_test ) ;
    end
end

%% Visulaization from 12th to 14th patient

if UseVisualisationCode == true
Patient12_Actual= dataset{12,1}.Image(:,:,:,5);
Patient13_Actual = dataset{13,1}.Image(:,:,:,5); % T2 image as reference
Patient14_Actual = dataset{14,1}.Image(:,:,:,5);
Patient12_Predicted= dataset{12,1}.Image(:,:,:,5);
Patient13_Predicted = dataset{13,1}.Image(:,:,:,5);
Patient14_Predicted = dataset{14,1}.Image(:,:,:,5); 
ValidationData=[ValidationData(:,1:11) labelPredicted(:,1)];
TestPatient12 = ValidationData(1:30656,:);
TestPatient13 = ValidationData(30657:52353,:);
TestPatient14 = ValidationData(52354:70027,:);
Predict_ID = 12;
 %Test Patient 12
for i = 1 : 30656
    if TestPatient12(i,7) == 1 % Expert label A :: Non Cancer prostate section
        Patient12_Actual(TestPatient12(i,9),TestPatient12(i,10),TestPatient12(i,11),1)= 0;
    elseif TestPatient12(i,7) == 2 % Expert label A :: Cancer prostate section
        Patient12_Actual(TestPatient12(i,9),TestPatient12(i,10),TestPatient12(i,11),1)= 1700;
    end
    if TestPatient12(i,Predict_ID) == 1 % Expert label A :: Non Cancer prostate section
        Patient12_Predicted(TestPatient12(i,9),TestPatient12(i,10),TestPatient12(i,11),1)= 0;
    elseif TestPatient12(i,Predict_ID) == 2 % Expert label A :: Cancer prostate section
        Patient12_Predicted(TestPatient12(i,9),TestPatient12(i,10),TestPatient12(i,11),1)= 1700;
    end
end
 %Test Patient 13
for i = 1 : 21696
    if TestPatient13(i,7) == 1 % Expert label A :: Non Cancer prostate section
        Patient13_Actual(TestPatient13(i,9),TestPatient13(i,10),TestPatient13(i,11),1)= 0;
    elseif TestPatient13(i,7) == 2 % Expert label A :: Cancer prostate section
        Patient13_Actual(TestPatient13(i,9),TestPatient13(i,10),TestPatient13(i,11),1)= 2500;
    end
    if TestPatient13(i,Predict_ID) == 1 % Expert label A :: Non Cancer prostate section
        Patient13_Predicted(TestPatient13(i,9),TestPatient13(i,10),TestPatient13(i,11),1)= 0;
    elseif TestPatient12(i,Predict_ID) == 2 % Expert label A :: Cancer prostate section
        Patient13_Predicted(TestPatient13(i,9),TestPatient13(i,10),TestPatient13(i,11),1)= 2500;
    end
end
 %Test Patient 14
for i = 1 : 17673
    if TestPatient12(i,7) == 1 % Expert label A :: Non Cancer prostate section
        Patient14_Actual(TestPatient14(i,9),TestPatient14(i,10),TestPatient14(i,11),1)= 0;
    elseif TestPatient12(i,7) == 2 % Expert label A :: Cancer prostate section
        Patient14_Actual(TestPatient14(i,9),TestPatient14(i,10),TestPatient14(i,11),1)= 2500;
    end
    if TestPatient12(i,Predict_ID) == 1 % Expert label A :: Non Cancer prostate section
        Patient14_Predicted(TestPatient14(i,9),TestPatient14(i,10),TestPatient14(i,11),1)= 0;
    elseif TestPatient12(i,Predict_ID) == 2 % Expert label A :: Cancer prostate section
        Patient14_Predicted(TestPatient14(i,9),TestPatient14(i,10),TestPatient14(i,11),1)= 2500;
    end
end


imagine(dataset{12,1}.Image(:,:,:,5),Patient12_Actual,Patient12_Predicted...
    ,dataset{13,1}.Image(:,:,:,5),Patient13_Actual,Patient13_Predicted...
    ,dataset{14,1}.Image(:,:,:,5),Patient14_Actual,Patient14_Predicted);
end






