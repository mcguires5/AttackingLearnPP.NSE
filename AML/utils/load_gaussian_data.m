%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script creating 2D gaussian function with respect to the parameters 
% given in the article Poisoning Attacks Against SVMs published in ICML 2012
%
% Input parameters : 
%   mu : scalar representing the mean of the distribution
%   sigma : scalar representing the standard deviation
%   num_samples_per_class : integer representing the number of data to be 
%                           created per class
%   num_feat : integer representing the dimensionality of the generated data
% Output parameters :
%   x : 2-column vector representing the data generated following a
%       gaussian distribution with respect to the given parameters
%   y : a 1D vector containing the associated labels 
%       (the first num_label_per_class data are labeled 1 and the others -1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [x y] = load_gaussian_data(mu,sigma,num_samples_per_class,num_feat)

% setting the parameter of the distribution
mu1= zeros(1,num_feat);
mu1(1)=-mu;
sigma1 = sigma*eye(num_feat);

mu2= zeros(1,num_feat);
mu2(1)=mu;
sigma2 = sigma*eye(num_feat);

% creates the gaussian data
x=mvnrnd(mu1,sigma1,num_samples_per_class);
x=[x; mvnrnd(mu2,sigma2,num_samples_per_class)];

y = [ones(num_samples_per_class,1); -ones(num_samples_per_class,1)];


