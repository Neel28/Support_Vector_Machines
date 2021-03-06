function CSCI567_hw3_spring16()
load test_x
load train_x
load train_y
load test_y

[train_scaled,test_scaled,accu_2,accu_3,c,accu_3t,accu_4,time_4,accupol,timep,cp,dp,accup,accurbf,cr,gr,accur] = main();
[bias,var,bias1,var1,res] = main6;

disp('Answer to 5.2, linear SVM = ')
accu_2

disp('Answer to 5.3 with different C values = ')
accu_3

disp('C value corresponding to best accuracy = ')
c

disp('Test Accuracy corresponding to above C value = ')
accu_3t

disp('Training Accuracy for linear kernel : = ')
accu_4

disp('Training Time for linear kernel : = ')
time_4

disp('Traning Accuracy for polynomial kernel : = ')
accupol

disp('Training Time for polynomial kernel : = ')
timep

disp('Best value of C and degree for best accuracy = ')
cp
dp

disp('Test data accuracy for above parameters = ')
accup

disp('Training Accuracy & Training Time for RBF kernel : = ')
accurbf

disp('Best value of C and gamma for best accuracy = ')
cr
gr

disp('Test data accuracy for above parameters = ')
accur

disp('Q 6')
disp('a. Bias = ')
bias
disp('a. Variance = ')
var
disp('b. Bias = ')
bias1
disp('b. Variance = ')
var1
disp('d.Bias and Variance = ')
res

end