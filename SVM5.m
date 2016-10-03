load('diabetic_test.mat');
testx = x; testy = y;

load('diabetic_train.mat');
trainx = x; trainy = y;

times = [];

accmatrix = zeros(3,9);

di = 1;
for degree = 1:3
    display(['Degree ', num2str(degree)]);
    ei = 1;
    for ex = -3:7
        C = 4^ex;
        args = ['-v 3 -q -t 1 -d ', num2str(degree), ' -c ', num2str(C)];
        tic;
        accuracy = svmtrain(trainy, trainx, args)
        accmatrix(round(di), round(ei)) = accuracy;
        time = toc;
        times = [times, time];
        ei = ei + 1;
    end
    di = di + 1;
end

figure,
%surf(1:3, -6:2, accmatrix);
bar3(accmatrix);
disp(['Average training time: ', num2str(mean(times))]);