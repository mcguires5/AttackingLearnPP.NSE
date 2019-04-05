clear;clc;
x=2.25;
m = 20;
xs=zeros(1,m);

taylorSeries = @(x,n) x.^n./factorial(n);
for n=0:m
    xs(n+1) = sum(taylorSeries(x,0:n));
    %percerrpr = abs(((exp(x)-sum(answer))))/exp(x)*100;
    
end
figure
plot([0 m], [9.4877, 9.4877])
hold on
plot(0:m,xs)