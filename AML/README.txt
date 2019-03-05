% Poisoning attacks against SVMs -- Matlab code for running the experiments reported in:
% 
% Battista Biggio, Blaine Nelson, and Pavel Laskov. Poisoning attacks against support vector machines.
% In J. Langford and J. Pineau, editors, 29th Int'€™l Conf. on Machine Learning. Omnipress, 2012.
% 
% http://pralab.diee.unica.it/en/node/729
% http://pralab.diee.unica.it/en/PoisoningAttacks
% 
% Copyright (C) 2013, Battista Biggio, Paul Temple. 
% Dept. of Electrical and Electronic Engineering, University of Cagliari, Italy.
% 
% This code is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This code is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.


This MATLAB code replicates the experiments reported in our paper
"Poisoning attacks against support vector machines" at ICML 2012.

You first need to download http://www.csie.ntu.edu.tw/~cjlin/libsvm/
and build the matlab wrapper for libSVM.
Then, place the corresponding mex files into the folder 'libsvm'.
Currently, only mex files for MAC OS X 10.8 are provided.

After having built the libSVM wrapper, just run main.m for the 2D toy experiments,
or main_MNIST.m for the experiments on the MNIST handwritten digits.

In this updated version 1.1 we corrected a bug pointed out by
Nathalie Baracaldo and Jaehoon Safavi
(IBM Almaden Research Center, San Jose, CA, USA)
and included the MNIST data as part of the package.



