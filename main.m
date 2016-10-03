function [train_scaled,test_scaled,accu_2,accu_3,c,accu_3t,accu_4,time_4,accupol,timep,cp,dp,accup,accurbf,cr,gr,accur] = main()
load test_x
load train_x
load train_y
load test_y

z = find(train_y==0);
train_y(z) = -1;
z = find(test_y==0);
test_y(z) = -1;

minimum = min(min(train_x),min(test_x));
maximum = max(max(train_x),max(test_x));
minimum=minimum';
maximum=maximum';
train_scaled = scaling(train_x,minimum,maximum);
test_scaled = scaling(test_x,minimum,maximum);

[w,eps,fval] = trainsvm(train_scaled,train_y,1);
accu_2 = testsvm(test_scaled,test_y,w)

[accu_3,c] = own_linear(train_scaled,train_y,test_scaled,test_y);
[w1,eps1,fval1] = trainsvm(train_scaled,train_y,c);
accu_3t = testsvm(test_scaled,test_y,w1)

[accu_4,time_4] = lib(train_scaled,train_y,test_scaled,test_y);

[accupol,timep,cp,dp,accup] = lib1(train_scaled,train_y,test_scaled,test_y);

[accurbf,cr,gr,accur] = lib2(train_scaled,train_y,test_scaled,test_y);
end

function [d] = scaling(data,minimum,maximum)
    d = zeros(size(data));
    [x,y] = size(data);
    for row=1:x
        for col=1:y
            d(row,col) = (data(row,col)-minimum(col,1))/(maximum(col,1)-minimum(col,1));
        end
    end
end

function [w,eps,fval] = trainsvm(x,y,C)

[Nthrow, Dthcol] = size(x);
s = Nthrow + Dthcol + 1;
x = [x, ones(Nthrow, 1)];
H = zeros(s);
H(1 : Dthcol, 1 : Dthcol) = eye(Dthcol);

A = zeros(2 * Nthrow, s);
A(1 : Nthrow, 1 : Dthcol + 1) = diag(y) * x;
A(1 : Nthrow, Dthcol + 2 : end) = eye(Nthrow);
A(Nthrow + 1 : end, Dthcol + 2 : end) = eye(Nthrow);

f = C * ones(s, 1);
f(1 : Dthcol + 1) = 0;

b = ones(2 * Nthrow, 1);
b(Nthrow + 1 : 2 * Nthrow) = 0;

A = -A; 
b = -b;
options = optimset('maxiter',2000);
[z, fval] = quadprog(H, f, A, b, [], [], [], [], [], options);
w = z(1 : Dthcol + 1);
eps = z(Dthcol + 2 : end);
end

function accu = testsvm(x,y,w)
b = w(end,1);
w = w(1:end-1,1);
margin = 1/norm(w);
res = sign(x*w + b*ones(size(x,1),1));
pos = length(find(res==y));
accu = pos*100/size(x,1);
end

function [res,c] = own_linear(x,y,X,Y)

C = [4^-6, 4^-5, 4^-4, 4^-3, 4^-2, 4^-1, 1, 4, 16];
res = zeros(length(C),2);
for c=1:length(C)
    res(c,:) = crossVal(x,y,X,Y,C(1,c)); 
end
%{
figure
subplot(1,2,1)
plot(C,res(:,1))
subplot(1,2,2)
plot(C,res(:,2))
%}
[x y] = max(res(:,1));
c=C(y);
end

function fin = crossVal(x,y,X,Y,C)
    indices = crossvalind('Kfold',y,3);
    accu = 0;
    tic;
    for i=1:3
        val=0;
        train=0;
        dtrain=0;
        ltrain=0;
        dval=0;
        lval=0;
    
        val = (indices == i); 
        train = ~val;
        dtrain = x(train,:);
        ltrain = y(train,:);
        dval = x(val,:);
        lval = y(val,:);
        
        [w,eps,fval] = trainsvm(dtrain,ltrain,C);
        accu = accu + testsvm(dval,lval,w); 
    end
    t = toc;
    accu = accu/3;
    time=t;
    fin=[accu time];
end

function [m,time] = lib(train_scaled,train_y,test_scaled,test_y)
    C = [4^-6, 4^-5, 4^-4,4^-3,4^-2,4^-1,1,4,16];
    time=zeros(length(C),1);
    m=[];
    for index=1:length(C)
        tic;
        model = svmtrain(train_y, train_scaled, ['-v 3 -t 0  -c ' ,num2str(C(index))]);
        time(index)=toc;
        m=[m;model];
    end         

end

function [x,time,c,d,accu] = lib1(train_scaled,train_y,test_scaled,test_y)
    
    C = [4^-3, 4^-2, 4^-1, 1, 4, 16,4^3,4^4,4^5,4^6,4^7];
    time=zeros(length(C),3);
    m=[];
    % For ploynomial kernel
    d=[1;2;3];
    for i=1:3
        for j=1:length(C)
            tic;
            model = svmtrain(train_y, train_scaled, ['-v 3 -t 1  -d ' ,num2str(d(i)),' -c ',num2str(C(j))]);
            time(j,i)=toc;
            m=[m;model];
        end
    end
    x=reshape(m,[],3);
    [p q]=max(x);
    q=q(1);
    c=C(q);
    d=find(max(p)==p);
    model = svmtrain(train_y, train_scaled, ['-t 1  -d ' ,num2str(d),' -c ',num2str(C)]);
    w_test = model.SVs' * model.sv_coef;
    b_test = -model.rho;
    r = sign(test_scaled*w_test + b_test*ones(size(test_scaled,1),1));
    p = length(find(r==test_y));
    accu = p*100/size(test_scaled,1) 
end

function [res,c,g,accu] = lib2(train_scaled,train_y,test_scaled,test_y)

    res = lib2_temp(train_scaled,train_y,test_scaled,test_y);
    [x y] = max(res(:,3));
    c = res(y,1);
    g = res(y,2);
    keyboard
    model = svmtrain(train_y, train_scaled, ['-t 2 -g ' ,num2str(g),' -c ',num2str(c)]);
    w_test = model.SVs' * model.sv_coef;
    b_test = -model.rho;
    r = sign(test_scaled*w_test + b_test*ones(size(test_scaled,1),1));
    p = length(find(r==test_y));
    accu = p*100/size(test_scaled,1) %61.88
    
    %plot
    ac=res(:,3);
    t1=res(:,4);
    acc=reshape(ac,[],11);
    acc=acc';
    t=reshape(t1,[],11);
    t=t';
    
    figure
    bar3(acc)
    xlabel('gamma')
    ylabel('C')
    zlabel('Accuracy')
    
    figure
    bar3(t)
    xlabel('gamma')
    ylabel('C')
    zlabel('Training Time')
    
end

function res = lib2_temp(train_scaled,train_y,test_scaled,test_y)
    
    C = [4^-3, 4^-2, 4^-1, 1, 4, 16,4^3,4^4,4^5,4^6,4^7];
    gamma = [4^-7, 4^-6,4^-5, 4^-4,4^-3, 4^-2, 4^-1, 1, 4, 16];
    m = length(train_y)
    [dist,d] = distMetric(train_scaled',train_scaled',1);
    gamma = gamma./d;
    m=[];
    time = [];
    carr = [];
    garr = [];

    for c=1:length(C)
        for g=1:length(gamma)
            tic;
            model = svmtrain(train_y, train_scaled, ['-v 3 -t 2 -g ' ,num2str(gamma(g)),' -c ',num2str(C(c))]);
            t=toc;
            m=[m;model];
            time=[time;t];
            carr=[carr;C(c)];
            garr=[garr;gamma(g)];
        end
    end
    res = [carr garr m time];
    
end







