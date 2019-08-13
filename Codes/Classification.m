
% before passing it to classification learner 
trainSubset = Sub6_Session1TOM(:,ranking(1:100));% subset of train data 
trainSubset = horzcat(trainSubset,Sub6_Session1TOM(:,2072));% get the class labels
testSubsetFinal = Sub6_Session2TOM(:,ranking(1:2071));% subset of test data 

%pass trainSubset to classification learner app and export model

% execute the below step in MATLAB command window to obtain accuracy of test data
%yfit_sub6_100 = trainedClassifier.predictFcn(testSubsetFinal);
%CP_sub6_100 = classperf(transpose(Sub1Session2Class), yfit_sub6_100); %gives accuracy on session 2