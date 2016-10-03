figure
subplot(2,3,1)
plot(C,acc(:,1))
xlabel('C')
ylabel('Accuracy')

subplot(2,3,2)
plot(C,acc(:,2))
xlabel('C')
ylabel('Accuracy')

subplot(2,3,3)
plot(C,acc(:,3))
xlabel('C')
ylabel('Accuracy')

subplot(2,3,4)
plot(C,time(:,1))
xlabel('C')
ylabel('Training time')

subplot(2,3,5)
plot(C,time(:,2))
xlabel('C')
ylabel('Training time')

subplot(2,3,6)
plot(C,time(:,3))
xlabel('C')
ylabel('Training time')

subplot(1,2,1)
bar3(res(:,1:3))
xlabel('C')
ylabel('Gamma')
zlabel('Accuracy')

subplot(1,2,2)
plot(C,time(:,3))
xlabel('C')
ylabel('Training time')

maximum = zeros(11,1)

figure
subplot(1,2,1)
bar3(x)
xlabel('gamma')
ylabel('C')
zlabel('Accuracy')

subplot(1,2,2)
bar3(time)
xlabel('gamma')
ylabel('C')
zlabel('Training time')

figure
subplot(1,2,1)
bar3(x)
xlabel('Degree')
ylabel('C')
zlabel('Accuracy')

subplot(1,2,2)
bar3(time)
xlabel('Degree')
ylabel('C')
zlabel('Training time')

figure
subplot(1,2,1)
plot(C,m)
xlabel('C')
ylabel('Accuracy')

subplot(1,2,2)
plot(C,time)
xlabel('C')
ylabel('Training Time')
