%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% display a squared image such as MNIST data images.
%
% Input parameter : 
%   c : a row vector representing the pixel of a squared image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function display_character(c)

% reshape the image c from row vector to squared image
imagesc(reshape(c,sqrt(numel(c)),sqrt(numel(c)))')
% display the image in grayscale
colormap gray