t = (0:0.001:1.5)';
data = 50*exp(-t).*cos(2*pi*t);
derivatives = -50*exp(-1*t).*(cos(2*pi*t)+2*pi*sin(2*pi*t));
Fprime = @(k,f,t) -1*exp(-1*k*t).*(k*cos(2*pi*f*t)+2*pi*f*sin(2*pi*f*t));
functions = "-e^(-1*k*t)*[k*cos(2*pi*f*t) + 2*pi*f*sin(2*pi*f*t)]";
functions  = ["1" repmat(functions,1,100)];
counter = 2;
Theta = zeros(length(t),100);
Theta(:,1) = 1;
for iK = 1:10
    for iF = 1:10
        Theta(:,counter) = Fprime(iK,iF,t);
        functions(1,counter) = strrep(functions(1,counter),'f',num2str(iF));
        functions(1,counter) = strrep(functions(1,counter),'k',num2str(iK));
        counter = counter + 1;
    end
end
functions = ["t" cellstr(functions)];
library = num2cell([t Theta]);
library = [functions;library];
Xi = sparsifyDynamics(Theta,derivatives,2e-6,1);
Xi = cellstr(num2str(Xi));
coefficients = cellstr([[" ","dx/dt"];[functions(2:end)',Xi]]);