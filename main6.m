function [bias,var,bias1,var1,res_d] = main6()
    % generating random data
    a = -1;
    b = 1;
    % part a data = 1000 samples
    data_x = zeros(10,100);
    data_y = zeros(10,100);
    for count=1:100
        r = (b-a).*rand(10,1) + a;
        noise = zeros(10,1);
        for z=1:10
            noise(z,1) = normrnd(0,0.1);
        end 
        op = 2*(r.^2) + noise;
        data_x(:,count) = r;
        data_y(:,count) = op;
    end
    % part b data = 10000 samples
    data_x1 = zeros(100,100);
    data_y1 = zeros(100,100);
    for count=1:100
        r = (b-a).*rand(100,1) + a;
        noise = zeros(100,1);
        for z=1:100
            noise(z,1) = normrnd(0,0.1);
        end 
        op = 2*(r.^2) + noise;
        data_x1(:,count) = r;
        data_y1(:,count) = op;
    end
    % a part
    [bias,var] = plot(data_x,data_y,0)
 
    % b part
    [bias1,var1] = plot(data_x1,data_y1,1)
    
    % d part
    lam = [0.01,0.1,1,10];
    res_d = zeros(4,2);
    for i=1:length(lam)
        b=zeros(100,1);
        v=zeros(100,1);
        for j=1:100
            [x,y] = plot1(data_x1(:,j),data_y1(:,j),100,lam(1,i));
            b(j,1) = x;
            v(j,1) = y;
        end
        res_d(i,1) = sum(b)/100;
        res_d(i,2) = sum(v)/100;
    end
end

function [r s] = sse1(x,y,send)
    m = length(y);
    w = 1;
    r = 0;
    r = sum((y - w).^2);
    s=0;
    r=r/send;
    s=s/send;
end

function [r s] = sse2(x,y,send)
    m = length(y);
    w0 = ones(m,1);
    w = glmfit(w0,y,'normal','constant','off');
    pred = glmval(w,w0,'identity','constant','off');
    r = 0;
    r = sum((y - pred).^2);
    mean_pred = mean(pred);
    s=0;
    s = sum((pred - mean_pred*ones(size(y,1),1)).^2);
    r=r/send;
    s=s/send;
end

function [r s] = sse3(x,y,send)
    m = length(y);
    x = [ones(m,1),x];
    w = glmfit(x,y,'normal','constant','off');
    pred = glmval(w,x,'identity','constant','off');
    r = 0;
    r = sum((y - pred).^2);
    mean_pred = mean(pred);
    s=0;
    s = sum((pred - mean_pred*ones(size(y,1),1)).^2);
    r=r/send;
    s=s/send;
end

function [r s] = sse4(x,y,send)
    m = length(y);
    x = [ones(m,1),x, x.^2];
    w = glmfit(x,y,'normal','constant','off');
    pred = glmval(w,x,'identity','constant','off');
    r = 0;
    r = sum((y - pred).^2);
    mean_pred = mean(pred);
    s=0;
    s = sum((pred - mean_pred*ones(size(y,1),1)).^2);
    r=r/send;
    s=s/send;
end

function [r s] = sse5(x,y,send)
    m = length(y);
    x = [ones(m,1),x,x.^2,x.^3];
    w = glmfit(x,y,'normal','constant','off');
    pred = glmval(w,x,'identity','constant','off');
    r = 0;
    r = sum((y - pred).^2);
    mean_pred = mean(pred);
    s=0;
    s = sum((pred - mean_pred*ones(size(y,1),1)).^2);
    r=r/send;
    s=s/send;
end
function [r s] = sse6(x,y,send)
    m = length(y);
    x = [ones(m,1),x,x.^2,x.^3,x.^4];
    w = glmfit(x,y,'normal','constant','off');
    pred = glmval(w,x,'identity','constant','off');
    r = 0;
    r = sum((y - pred).^2);
    mean_pred = mean(pred);
    s=0;
    s = sum((pred - mean_pred*ones(size(y,1),1)).^2);
    r=r/send;
    s=s/send;
end

function [bias,var] = plot(data_x,data_y,col)
    if col==0
        send=10;
    elseif col==1
        send=100;
    end
    col=100;
    bias=zeros(6,1);
    var=zeros(6,1);
    figure
    val = zeros(col,2);
    % g1(x)
    for i=1:col
        [p,q] = sse1(data_x(:,i),data_y(:,i),send);
        val(i,1) = p;
        val(i,2) = q;
    end
    bias(1,1)= sum(val(:,1))/100;
    var(1,1) = sum(val(:,2))/100;
    subplot(2,3,1)
    hist(val(:,1))
    
    val = zeros(col,2);
    % g2(x)
    for i=1:col
        [p,q] = sse2(data_x(:,i),data_y(:,i),send);
        val(i,1) = p;
        val(i,2) = q;
    end
    bias(2,1)= sum(val(:,1))/100;
    var(2,1) = sum(val(:,2))/100;
    subplot(2,3,2)
    hist(val(:,1))
    
    val = zeros(col,2);
    % g3(x)
    for i=1:col
        [p,q] = sse3(data_x(:,i),data_y(:,i),send);
        val(i,1) = p;
        val(i,2) = q;
    end
    bias(3,1)= sum(val(:,1))/100;
    var(3,1) = sum(val(:,2))/100;
    subplot(2,3,3)
    hist(val(:,1))
    
    val = zeros(col,2);
    % g4(x)
    for i=1:col
        [p,q] = sse4(data_x(:,i),data_y(:,i),send);
        val(i,1) = p;
        val(i,2) = q;
    end
    bias(4,1)= sum(val(:,1))/100;
    var(4,1) = sum(val(:,2))/100;
    subplot(2,3,4)
    hist(val(:,1))
    
    val = zeros(col,2);
    % g5(x)
    for i=1:col
       [p,q] = sse5(data_x(:,i),data_y(:,i),send);
       val(i,1) = p;
       val(i,2) = q;
    end
    bias(5,1)= sum(val(:,1))/100;
    var(5,1) = sum(val(:,2))/100;
    subplot(2,3,5)
    hist(val(:,1))
    
    val = zeros(col,2);
    % g6(x)
    for i=1:col
        [p,q] = sse6(data_x(:,i),data_y(:,i),send);
        val(i,1) = p;
        val(i,2) = q;
    end
    bias(6,1)= sum(val(:,1))/100;
    var(6,1) = sum(val(:,2))/100;
    subplot(2,3,6)
    hist(val(:,1))
end

function [r,s] = plot1(x,y,col,lam,w)
    m = length(y);
    x = [ones(m,1),x,x.^2];
    w = pinv(x'*x + (lam*m*eye(3)))*x'*y;
    pred=zeros(100,1);
    for c=1:100
        pred(c,1) = x(c,:)*w;
    end
    r = 0;
    r = sum((y - pred).^2);
    mean_pred = mean(pred);
    s=0;
    s = sum((pred - mean_pred*ones(size(y,1),1)).^2);
    r=r/100;
    s=s/100;
end





