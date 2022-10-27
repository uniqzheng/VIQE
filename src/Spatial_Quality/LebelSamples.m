function [TrainLable TestLable]=LebelSamples(Num_Class, No_TrainSamples, No_TestSamples)
TrainLable1=[]; TestLable1=[];
for i=1: Num_Class
    Ltrain=repmat(i,No_TrainSamples); Ltest=repmat(i,No_TestSamples);
    Ltrain1=Ltrain(1,:); Ltest1=Ltest(1,:);
    TrainLable1=[TrainLable1 Ltrain1]; TestLable1=[TestLable1 Ltest1];
end
TrainLable=TrainLable1';
TestLable=TestLable1';
