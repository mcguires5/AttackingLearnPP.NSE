% @Author: Delengowski
% @Date:   2019-01-27 18:54:05
% @Last Modified by:   delengowski
% @Last Modified time: 2019-03-06 18:16:16

sindy = SINDy();
sindy.lambda = 2e-4;
sindy.polyOrder = 5;
%sindy.useSine = 1;
%sindy.useExp = 1;
%sindy.sineMultiplier = 4;
%sindy.expMultiplier = 3;
sindy.buildModel([(1:.01:10)'.^2 (1:.01:10)'.^2],.01,1,10,50);
sindy.displayLibrary;
sindy.displayModel('stacked');